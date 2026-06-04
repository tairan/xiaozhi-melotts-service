from pydantic import BaseModel, Field, model_validator
from typing import Optional

class TTSRequest(BaseModel):
    text: Optional[str] = Field(None, description="The text to synthesize.")
    input: Optional[str] = Field(None, description="Alternative text field (alias for compatibility).")
    language: str = Field("ZH", description="The model language code (e.g. ZH, EN, JP, KR, ES, FR).")
    speaker: Optional[str] = Field(None, description="The speaker name or ID within the selected language model. Defaults to first speaker.")
    voice: Optional[str] = Field(None, description="Alternative speaker name (alias for compatibility).")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Inference speed multiplier.")
    stream: bool = Field(False, description="Whether to stream the audio chunks sentence-by-sentence.")
    format: str = Field("wav", description="Audio format output. Supported: wav, mp3, pcm.")
    output_dir: Optional[str] = Field(None, description="Output directory to save audio files on the server.")

    @model_validator(mode='before')
    @classmethod
    def resolve_aliases(cls, data):
        if isinstance(data, dict):
            # Fallback text/input
            if not data.get('text'):
                if data.get('input'):
                    data['text'] = data['input']
            # Fallback speaker/voice
            if not data.get('speaker'):
                if data.get('voice'):
                    data['speaker'] = data['voice']
        return data

class OpenAISpeechRequest(BaseModel):
    model: str = Field("tts-1", description="Model name. Accepts tts-1 or tts-1-hd.")
    input: Optional[str] = Field(None, description="The text to generate audio for.")
    text: Optional[str] = Field(None, description="Alternative text field (alias for compatibility).")
    voice: Optional[str] = Field(None, description="The voice to use. Accepts alloy, echo, fable, onyx, nova, shimmer, or raw speaker names.")
    speaker: Optional[str] = Field(None, description="Alternative speaker name (alias for compatibility).")
    response_format: str = Field("mp3", description="The format to audio output. Supported: mp3, wav, pcm.")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="The speed of the generated audio.")
    output_dir: Optional[str] = Field(None, description="Output directory to save audio files on the server.")

    @model_validator(mode='before')
    @classmethod
    def resolve_aliases(cls, data):
        if isinstance(data, dict):
            # Fallback input/text
            if not data.get('input'):
                if data.get('text'):
                    data['input'] = data['text']
            # Fallback voice/speaker
            if not data.get('voice'):
                if data.get('speaker'):
                    data['voice'] = data['speaker']
        return data
