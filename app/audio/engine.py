import numpy as np
import time
import torch

from threading import Event, Thread

from .audio_sample import EmptySampleException
from .audio_utils import read_audio_sample
from .player import Player
from .rave import load_model

class NotifyingEvent(Event):
    def __init__(self, callback, *a, **k):
        super().__init__(*a, **k)
        self._callback = callback

    def set(self, *a, **k):
        super().set(*a, **k)
        self._callback()

class AudioEngine(Thread):

    # def __init__(self, input_wav, model_path, latent_coordinates, *a, **k):
    def __init__(self, input_wav, model_path, *a, **k):
        super().__init__(*a, **k)

        sample = read_audio_sample(input_wav)
        if len(sample.data) <= 0:
            raise EmptySampleException
        self._sample = sample
        self._data_type = sample.data_type

        self._buffer_size = int(sample.sampling_rate / 4)

        self._player = Player(
            generate_data_callback=self._generate_next_buffer,
            sample_data_type=sample.data_type,
            number_of_channels=sample.number_of_channels,
            sample_rate=sample.sampling_rate,
            buffer_size=self._buffer_size,
        )

        self._keep_alive = True

        self._max_position = len(sample.data) - 1
        self._current_position = 0

        self._zeros = np.zeros(self._buffer_size, dtype=self._data_type)

        self._rave_model = load_model(model_path)
        if sample.sampling_rate != self._rave_model.sampling_rate:
            print(f"sampling rates differ")
            # TODO: resample the input sample

        self._latent_coordinates = torch.zeros(self._rave_model.num_latent_dimensions)
        # self._latent_coordinates = latent_coordinates
        self._rave_warmup()

        self.loop = Event()
        self.transform = Event()
        self.stop = NotifyingEvent(self._reset_engine_state)
        self.shutdown = NotifyingEvent(self._stop_engine)

    def _rave_warmup(self):
        for _ in range(4):
            self._apply_transformation(self._sample.data[:self._buffer_size])

    def _reset_engine_state(self):
        self._current_position = 0
        # self._latent_coordinates = torch.zeros(self._rave_model.num_latent_dimensions)

    def _stop_engine(self):
        self._keep_alive = False

    def _apply_transformation(self, buffer):
        torch.set_grad_enabled(False)
        input_data = torch.Tensor(buffer)
        input_data = torch.reshape(input_data, (1, 1, buffer.size))
        encoded = self._rave_model.encode(input_data)
        encoded[0][:,0] += self._latent_coordinates
        decoded = self._rave_model.decode(encoded)
        return decoded[0][0][:buffer.size].numpy()

    def _generate_next_sample_buffer(self):
        data = self._sample.data
        for _ in range(self._buffer_size):
            position = self._current_position
            if position < self._max_position:
                yield data[position]
                self._current_position += 1
            else:
                yield 0

    def _generate_next_sample_buffer_looping(self):
        data = self._sample.data
        for _ in range(self._buffer_size):
            position = self._current_position % self._max_position
            yield data[position]
            self._current_position = position + 1

    def _generate_next_buffer(self):
        if self.stop.is_set():
            return self._zeros

        sample_buffer = np.fromiter(
            self._generate_next_sample_buffer_looping()
                if self.loop.is_set()
                else self._generate_next_sample_buffer(),
            dtype=float,
            count=self._buffer_size,
        )
        if self.transform.is_set():
            sample_buffer = self._apply_transformation(sample_buffer)
        return sample_buffer.astype(self._data_type)

    @property
    def latent_coordinates(self):
        return self._latent_coordinates

    # TODO: make thread safe (use Python Queue?)
    def set_latent_coordinates(self, coordinates):
        num_dimensions = len(self._latent_coordinates)
        num_coordinates = len(coordinates)
        if num_coordinates > num_dimensions:
            coordinates = coordinates[:num_dimensions]
        self._latent_coordinates[:num_coordinates] = torch.Tensor(coordinates)

    def run(self):
        with self._player.stream_audio():
            while self._keep_alive:
                time.sleep(0.1)
