import sounddevice as sd
import wave

from .audio_sample import AudioSample

def read_audio_sample(path):
    with wave.open(path, "rb") as f:
        fs = f.getframerate()
        channels_no = f.getnchannels()
        data = f.readframes(-1)
        sample_width = f.getsampwidth()
        return AudioSample(fs, channels_no, data, sample_width)

def play(audio_data, fs):
    sd.play(audio_data, fs)
