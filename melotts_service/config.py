import os

# Set mirror for HF models
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Global output directory configuration
GLOBAL_OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", None)

# Model configuration mappings from ModelScope
MODEL_MAPPING = {
    'EN': 'myshell-ai/MeloTTS-English',
    'EN_V2': 'myshell-ai/MeloTTS-English-v2',
    'FR': 'myshell-ai/MeloTTS-French',
    'JP': 'myshell-ai/MeloTTS-Japanese',
    'ES': 'myshell-ai/MeloTTS-Spanish',
    'ZH': 'myshell-ai/MeloTTS-Chinese',
    'KR': 'myshell-ai/MeloTTS-Korean',
}

# Force CPU device due to host PyTorch/sm_61 compatibility
DEFAULT_DEVICE = 'cpu'
