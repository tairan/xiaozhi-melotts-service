import os
import re
import io
import torch
import struct
import asyncio
import uuid
import time
import numpy as np
import soundfile
from typing import Optional, List
from fastapi import HTTPException, status
from modelscope import snapshot_download
from melo.api import TTS
from melo import utils
from starlette.concurrency import run_in_threadpool

from melotts_service.config import MODEL_MAPPING, DEFAULT_DEVICE, GLOBAL_OUTPUT_DIR

_loaded_models = {}
_model_lock = asyncio.Lock()

def get_or_load_model(language_code: str, device: str = 'auto') -> TTS:
    """Synchronous model loader that runs inside a thread pool."""
    lang = language_code.upper()
    if lang == 'ZH_MIX_EN':
        lang = 'ZH'
    
    if lang not in MODEL_MAPPING:
        short_lang = lang.split('-')[0]
        if short_lang in MODEL_MAPPING:
            lang = short_lang
        else:
            raise ValueError(f"Language {language_code} is not supported.")
            
    if device == 'auto':
        device = DEFAULT_DEVICE
        
    cache_key = (lang, device)
    if cache_key in _loaded_models:
        return _loaded_models[cache_key]
        
    # Download from ModelScope
    repo_id = MODEL_MAPPING[lang]
    print(f"Downloading model '{repo_id}' from ModelScope...")
    model_dir = snapshot_download(repo_id)
    print(f"Model downloaded successfully to: {model_dir}")
    
    config_path = os.path.join(model_dir, "config.json")
    ckpt_path = os.path.join(model_dir, "checkpoint.pth")
    
    # Check if checkpoint exists, otherwise find first .pth file
    if not os.path.exists(ckpt_path):
        files = os.listdir(model_dir)
        pth_files = [f for f in files if f.endswith(".pth")]
        if pth_files:
            ckpt_path = os.path.join(model_dir, pth_files[0])
        else:
            raise FileNotFoundError(f"No .pth file found in {model_dir}")
            
    print(f"Loading MeloTTS model for '{lang}' on device '{device}'...")
    model = TTS(
        language=lang,
        device=device,
        config_path=config_path,
        ckpt_path=ckpt_path
    )
    _loaded_models[cache_key] = model
    return model

async def get_or_load_model_async(language_code: str, device: str = 'auto') -> TTS:
    """Async wrapper that protects the model dictionary with an asyncio lock."""
    lang = language_code.upper()
    if lang == 'ZH_MIX_EN':
        lang = 'ZH'
        
    if lang not in MODEL_MAPPING:
        short_lang = lang.split('-')[0]
        if short_lang in MODEL_MAPPING:
            lang = short_lang
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Language '{language_code}' is not supported. Supported: {list(MODEL_MAPPING.keys())}"
            )
            
    if device == 'auto':
        device = DEFAULT_DEVICE
        
    cache_key = (lang, device)
    async with _model_lock:
        if cache_key in _loaded_models:
            return _loaded_models[cache_key]
            
        def _load():
            return get_or_load_model(lang, device)
            
        model = await run_in_threadpool(_load)
        _loaded_models[cache_key] = model
        return model

def get_speaker_id(model: TTS, speaker_name: Optional[str] = None) -> int:
    """Maps a speaker name (string) to an integer ID using the model's configuration."""
    speaker_ids = model.hps.data.spk2id
    if not speaker_name:
        return list(speaker_ids.values())[0]
        
    # Direct match check
    if speaker_name in speaker_ids:
        return speaker_ids[speaker_name]
        
    # Case insensitive check
    for k, v in speaker_ids.items():
        if k.lower() == speaker_name.lower():
            return v
            
    # Check if input is a valid integer string representing an ID
    try:
        spk_id = int(speaker_name)
        if spk_id in speaker_ids.values():
            return spk_id
    except ValueError:
        pass
        
    # Fallback to the first speaker
    return list(speaker_ids.values())[0]

def get_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Generates a standard 44-byte WAV header for streaming purposes.
    Sets the data length to a very large number to signal a live stream.
    """
    file_size = 0xFFFFFFFF
    data_size = file_size - 36
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        file_size,
        b'WAVE',
        b'fmt ',
        16,            # Subchunk1Size (16 for PCM)
        1,             # AudioFormat (1 for PCM)
        num_channels,  # NumChannels
        sample_rate,   # SampleRate
        sample_rate * num_channels * (bits_per_sample // 8),  # ByteRate
        num_channels * (bits_per_sample // 8),                # BlockAlign
        bits_per_sample,                                      # BitsPerSample
        b'data',
        data_size
    )
    return header

async def generate_audio_stream(text: str, model: TTS, speaker_id: int, speed: float = 1.0, format: str = "wav", output_dir: str = None):
    """Async generator that yields audio chunks sentence-by-sentence to support streaming and optionally saves a copy."""
    filepath = None
    file_handle = None
    
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.{format}"
            filepath = os.path.join(output_dir, filename)
            file_handle = open(filepath, "wb")
            print(f"Streaming output copy: writing to {filepath}")
        except Exception as e:
            print(f"Failed to create output file for streaming copy: {e}")

    language = model.language
    # Split the input text into individual sentences
    texts = model.split_sentences_into_pieces(text, language, quiet=True)
    
    # If WAV is requested, stream the WAV header first
    if format == "wav":
        header = get_wav_header(model.hps.data.sampling_rate, num_channels=1, bits_per_sample=16)
        if file_handle:
            file_handle.write(header)
        yield header
        
    device = model.device
    for t in texts:
        if language in ['EN', 'ZH_MIX_EN']:
            # Adjust word separations
            t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            
        def _infer():
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
                t, language, model.hps, device, model.symbol_to_id
            )
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio_arr = model.model.infer(
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    sdp_ratio=0.2,
                    noise_scale=0.6,
                    noise_scale_w=0.8,
                    length_scale=1. / speed,
                )[0][0, 0].data.cpu().float().numpy()
                
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
            return audio_arr
            
        # Run inference in threadpool to keep the FastAPI main loop responsive
        audio = await run_in_threadpool(_infer)
        
        # Convert audio array to 16-bit PCM bytes
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()
        
        if format in ["wav", "pcm"]:
            if file_handle:
                file_handle.write(pcm_bytes)
            yield pcm_bytes
        elif format == "mp3":
            def _to_mp3():
                from pydub import AudioSegment
                import io
                segment = AudioSegment(
                    data=pcm_bytes,
                    sample_width=2,
                    frame_rate=model.hps.data.sampling_rate,
                    channels=1
                )
                out_fp = io.BytesIO()
                segment.export(out_fp, format="mp3")
                return out_fp.getvalue()
                
            mp3_bytes = await run_in_threadpool(_to_mp3)
            if file_handle:
                file_handle.write(mp3_bytes)
            yield mp3_bytes
            
    if file_handle:
        file_handle.close()
        # If WAV, fix sizes in the header to ensure validity
        if format == "wav" and filepath and os.path.exists(filepath):
            try:
                file_len = os.path.getsize(filepath)
                with open(filepath, "r+b") as f:
                    f.seek(4)
                    f.write(struct.pack('<I', file_len - 8))
                    f.seek(40)
                    f.write(struct.pack('<I', file_len - 44))
            except Exception as e:
                print(f"Failed to fix stream output WAV header sizes: {e}")
                
    if torch.cuda.is_available():
        await run_in_threadpool(torch.cuda.empty_cache)

async def generate_complete_audio(text: str, model: TTS, speaker_id: int, speed: float = 1.0, format: str = "wav", output_dir: str = None) -> bytes:
    """Helper to generate and encode the complete audio without streaming, and optionally save a copy."""
    language = model.language
    texts = model.split_sentences_into_pieces(text, language, quiet=True)
    audio_list = []
    device = model.device
    
    for t in texts:
        if language in ['EN', 'ZH_MIX_EN']:
            t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            
        def _infer():
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
                t, language, model.hps, device, model.symbol_to_id
            )
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio_arr = model.model.infer(
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    sdp_ratio=0.2,
                    noise_scale=0.6,
                    noise_scale_w=0.8,
                    length_scale=1. / speed,
                )[0][0, 0].data.cpu().float().numpy()
                
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
            return audio_arr
            
        audio = await run_in_threadpool(_infer)
        audio_list.append(audio)
        
    if torch.cuda.is_available():
        await run_in_threadpool(torch.cuda.empty_cache)
        
    # Concatenate sentence audio segments
    audio_concat = model.audio_numpy_concat(audio_list, sr=model.hps.data.sampling_rate, speed=speed)
    audio_concat = np.clip(audio_concat, -1.0, 1.0)
    
    def _encode():
        out_fp = io.BytesIO()
        if format == "wav":
            soundfile.write(out_fp, audio_concat, model.hps.data.sampling_rate, format="WAV")
        elif format == "mp3":
            # Write to temporary WAV in memory, then load in pydub and export to MP3
            wav_fp = io.BytesIO()
            soundfile.write(wav_fp, audio_concat, model.hps.data.sampling_rate, format="WAV")
            wav_fp.seek(0)
            from pydub import AudioSegment
            segment = AudioSegment.from_wav(wav_fp)
            segment.export(out_fp, format="mp3")
        elif format == "pcm":
            audio_int16 = (audio_concat * 32767).astype(np.int16)
            out_fp.write(audio_int16.tobytes())
        return out_fp.getvalue()
        
    encoded_bytes = await run_in_threadpool(_encode)
    
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.{format}"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "wb") as f:
                f.write(encoded_bytes)
            print(f"Saved complete audio to {filepath}")
        except Exception as e:
            print(f"Failed to save complete audio copy: {e}")
            
    return encoded_bytes

def map_openai_voice(voice_name: str) -> tuple[str, str]:
    """Maps OpenAI standard voices to MeloTTS model and speaker names.
    Also handles passing of custom voices dynamically.
    """
    v = voice_name.lower()
    if v == 'alloy':
        return 'EN', 'EN-Default'
    elif v == 'echo':
        return 'EN', 'EN-BR'
    elif v == 'fable':
        return 'EN', 'EN-AU'
    elif v == 'onyx':
        return 'EN', 'EN_INDIA'
    elif v == 'nova':
        return 'EN', 'EN-Default'
    elif v == 'shimmer':
        return 'EN_V2', 'EN-US'
        
    # Shortcut language mappings
    elif v == 'zh':
        return 'ZH', 'ZH'
    elif v == 'jp':
        return 'JP', 'JP'
    elif v == 'kr':
        return 'KR', 'KR'
    elif v == 'es':
        return 'ES', 'ES'
    elif v == 'fr':
        return 'FR', 'FR'
        
    # Check if voice specifies speaker code directly
    if v.startswith('en-') or v.startswith('en_'):
        return 'EN', voice_name.upper()
    elif v.startswith('zh-') or v.startswith('zh_'):
        return 'ZH', 'ZH'
    elif v.startswith('jp-') or v.startswith('jp_'):
        return 'JP', 'JP'
    elif v.startswith('kr-') or v.startswith('kr_'):
        return 'KR', 'KR'
    elif v.startswith('es-') or v.startswith('es_'):
        return 'ES', 'ES'
    elif v.startswith('fr-') or v.startswith('fr_'):
        return 'FR', 'FR'
        
    # Fallback to Chinese model
    return 'ZH', 'ZH'
