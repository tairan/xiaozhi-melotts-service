import os
import io
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from melotts_service.config import GLOBAL_OUTPUT_DIR, MODEL_MAPPING
from melotts_service.schemas import TTSRequest, OpenAISpeechRequest
from melotts_service.core import (
    get_or_load_model_async, get_speaker_id,
    generate_audio_stream, generate_complete_audio, map_openai_voice
)

app = FastAPI(
    title="MeloTTS API & Streaming Service",
    description="A high-performance text-to-speech service based on MeloTTS, using ModelScope for model distribution, featuring OpenAI-compatible API and live streaming responses.",
    version="0.1.0"
)

# Enable CORS for all origins (useful for web interfaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Serves the interactive web portal for MeloTTS."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>MeloTTS Service Online</h1>")

@app.get("/voices")
async def get_voices():
    """Returns a list of all available languages and their speaker options."""
    voices_info = []
    # Dynamic values mapping for documented voices
    speakers_list = {
        'EN': ['EN-Default', 'EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU'],
        'EN_V2': ['EN-Default', 'EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU'],
        'FR': ['FR'],
        'JP': ['JP'],
        'ES': ['ES'],
        'ZH': ['ZH', 'EN', 'ZH_MIX_EN'],
        'KR': ['KR']
    }
    for lang, repo in MODEL_MAPPING.items():
        voices_info.append({
            "language": lang,
            "repo_id": repo,
            "speakers": speakers_list.get(lang, [])
        })
    return {"voices": voices_info}

@app.get("/tts")
async def custom_tts_get(
    text: str,
    language: str = "ZH",
    speaker: Optional[str] = None,
    speed: float = 1.0,
    stream: bool = False,
    format: str = "wav",
    output_dir: Optional[str] = Query(None, description="Output directory to save audio files on the server.")
):
    """GET endpoint to support streaming in HTML5 <audio> players directly via URL."""
    model = await get_or_load_model_async(language)
    speaker_id = get_speaker_id(model, speaker)
    format = format.lower()
    
    if format not in ["wav", "mp3", "pcm"]:
        raise HTTPException(status_code=400, detail="Invalid format. Supported formats: wav, mp3, pcm.")
        
    mime_types = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "pcm": "audio/pcm"
    }
    
    target_output_dir = output_dir or GLOBAL_OUTPUT_DIR
    if stream:
        generator = generate_audio_stream(text, model, speaker_id, speed, format, output_dir=target_output_dir)
        return StreamingResponse(generator, media_type=mime_types[format])
    else:
        audio_data = await generate_complete_audio(text, model, speaker_id, speed, format, output_dir=target_output_dir)
        return StreamingResponse(io.BytesIO(audio_data), media_type=mime_types[format])

@app.post("/tts")
async def custom_tts(req: TTSRequest):
    """General purpose Text-to-Speech API. Supports streaming and complete file return."""
    # Resolve aliases manually in case validation bypassed or for explicit check
    text = req.text or req.input
    if not text:
        raise HTTPException(status_code=400, detail="Missing text parameter. Specify either 'text' or 'input'.")
        
    speaker = req.speaker or req.voice
    
    model = await get_or_load_model_async(req.language)
    speaker_id = get_speaker_id(model, speaker)
    format = req.format.lower()
    
    if format not in ["wav", "mp3", "pcm"]:
        raise HTTPException(status_code=400, detail="Invalid format. Supported formats: wav, mp3, pcm.")
        
    mime_types = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "pcm": "audio/pcm"
    }
    
    target_output_dir = req.output_dir or GLOBAL_OUTPUT_DIR
    if req.stream:
        generator = generate_audio_stream(text, model, speaker_id, req.speed, format, output_dir=target_output_dir)
        return StreamingResponse(generator, media_type=mime_types[format])
    else:
        audio_data = await generate_complete_audio(text, model, speaker_id, req.speed, format, output_dir=target_output_dir)
        return StreamingResponse(io.BytesIO(audio_data), media_type=mime_types[format])

@app.post("/v1/audio/speech")
async def openai_speech(req: OpenAISpeechRequest):
    """OpenAI compatible text-to-speech API endpoint. Works with third-party applications."""
    text = req.input or req.text
    if not text:
        raise HTTPException(status_code=400, detail="Missing input text parameter. Specify either 'input' or 'text'.")
        
    voice = req.voice or req.speaker or "alloy"
    
    lang, speaker_name = map_openai_voice(voice)
    model = await get_or_load_model_async(lang)
    speaker_id = get_speaker_id(model, speaker_name)
    format = req.response_format.lower()
    
    if format not in ["mp3", "wav", "pcm"]:
        raise HTTPException(status_code=400, detail="Unsupported response format. Supported: mp3, wav, pcm.")
        
    mime_types = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "pcm": "audio/pcm"
    }
    
    target_output_dir = req.output_dir or GLOBAL_OUTPUT_DIR
    # We yield streaming response for speech endpoint since it provides the best latency
    generator = generate_audio_stream(text, model, speaker_id, req.speed, format, output_dir=target_output_dir)
    return StreamingResponse(generator, media_type=mime_types[format])
