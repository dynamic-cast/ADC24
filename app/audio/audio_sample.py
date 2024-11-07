import numpy as np


SAMPLE_WIDTH_TO_NP_DATA_TYPE = {
    1: np.int8,
    2: np.int16,
    4: np.float32, # np.int32,
}

class EmptySampleException(Exception):
    def __str__(self):
        return "Please provide a non-empty sample"

class UnsupportedSampleWidth(Exception):
    def __str__(self):
        return "Can't read the sample because of unsupported sample width"

def get_dtype_from_width(width):
    if width == 1:
        return np.uint8
    elif width == 2:
        return np.int16
    elif width == 4:
        return np.float32
    else:
        raise UnsupportedSampleWidth()

class AudioSample(object):
    """A read-only container for audio sample data"""
    def __init__(self, fs, channels_no, raw_data, sample_width):
        self._fs = fs
        # self._channels_no = channels_no
        # Mono only (for now...)
        self._channels_no = 1

        # An array with frames of audio sample data.
        # A frame size depends on the number of channels and
        # the with of an audio sample in bytes:
        #   frame_size = channels_no * sample_width
        try:
            self._data_type = SAMPLE_WIDTH_TO_NP_DATA_TYPE[sample_width]
        except KeyError:
            raise UnsupportedSampleWidth()

        data = np.fromstring(raw_data, self._data_type)
        # Normalize the data
        data = data / np.max(np.abs(data), axis=0)
        if channels_no > 1:
            data = np.resize(data, (int(data.size / channels_no), channels_no))
            # Turn into mono
            data = data[:,0]

        self._data = data

        self._sample_width = sample_width

    @property
    def sampling_rate(self):
        """Sampling rate in Hz"""
        return self._fs

    @property
    def number_of_channels(self):
        """Number of audio channels. Returns 1 for mono, 2 for stereo"""
        return self._channels_no

    @property
    def data(self):
        """A numpy array of frames of audio data"""
        return self._data

    @property
    def sample_width(self):
        """Sample width in bytes"""
        return self._sample_width

    @property
    def data_type(self):
        """Numpy data type of the stored sample data"""
        # return self._data_type
        # After normalization
        return np.float32
