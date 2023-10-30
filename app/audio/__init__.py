from .engine import AudioEngine

def create_audio_engine(input_wav, model_path):
    return AudioEngine(input_wav, model_path)
