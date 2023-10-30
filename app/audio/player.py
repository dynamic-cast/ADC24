import sounddevice
import numpy as np

class Player(object):
    def __init__(
        self,
        generate_data_callback,
        sample_data_type,
        number_of_channels,
        sample_rate,
        buffer_size,
        *a,
        **k
    ):
        self._generate_data = generate_data_callback
        self._number_of_channels = number_of_channels
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size
        self._dtype = sample_data_type

    def stream_audio(self):
        def callback(outdata, frames, time, status):
            data = self._generate_data()
            if data.ndim < 2: # bodge?
                data = data.reshape(-1, 1)
            outdata[:] = data

        return sounddevice.OutputStream(
            channels=self._number_of_channels,
            callback=callback,
            samplerate=self._sample_rate,
            dtype=self._dtype,
            blocksize=self._buffer_size,
        )
