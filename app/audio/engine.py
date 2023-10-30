import numpy as np
import time

from threading import Event, Thread

from .audio_sample import EmptySampleException
from .audio_utils import read_audio_sample
from .player import Player

class AudioEngine(Thread):

    def __init__(self, input_wav, *a, **k):
        super().__init__(*a, **k)

        sample = read_audio_sample(input_wav)
        if len(sample.data) <= 0:
            raise EmptySampleException
        self._sample = sample
        self._data_type = sample.data_type

        # self._buffer_size = int(sample.sampling_rate / 4)
        self._buffer_size = 32

        self._player = Player(
            generate_data_callback=self._generate_next_buffer,
            sample_data_type=sample.data_type,
            number_of_channels=sample.number_of_channels,
            sample_rate=sample.sampling_rate,
            buffer_size=self._buffer_size,
        )

        self._max_position = len(sample.data) - 1
        self._current_position = 0.0

        # self._zeros = np.zeros(self._buffer_size, dtype=self._data_type)

        self.stop = Event()

    def _generate_next_sample_buffer(self):
        # TODO: Add looping
        data = self._sample.data
        for _ in range(self._buffer_size):
            position = self._current_position
            if position < self._max_position:
                yield data[int(position)]
                self._current_position += 1.0
            else:
                yield 0

    def _generate_next_buffer(self):
        sample_buffer = np.fromiter(
            self._generate_next_sample_buffer(),
            dtype=float,
            count=self._buffer_size,
        )
        return sample_buffer.astype(self._data_type)

    def run(self):
        with self._player.stream_audio():
            while not self.stop.is_set():
                time.sleep(0.1)
