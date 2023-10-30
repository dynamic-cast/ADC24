import numpy as np
import torch

from .audio_utils import play, read_audio_sample

class RaveModelRepresentation():
    def __init__(self, module, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._module = module

        self._config = {
            "sampling_rate": 44100,
            "latent_size": 0,
            "encode_params": torch.Tensor(),
            "decode_params": torch.Tensor(),
            "prior_params": torch.Tensor(),
        }

        config_params = self._config.keys()
        for name, val in module.named_buffers():
            if name in config_params:
                self._config[name] = val

    @property
    def num_latent_dimensions(self):
        params = self._config['decode_params']
        return int(params[0]) if len(params) > 0 else 0

    @property
    def sampling_rate(self):
        return int(self._config["sampling_rate"])

    def encode(self, *a, **kw):
        return self._module.encode(*a, **kw)

    def decode(self, *a, **kw):
        return self._module.decode(*a, **kw)

    def forward(self, *a, **kw):
        return self._module.forward(*a, **kw)

def load_model(path):
    script_module = torch.jit.load(path)
    return RaveModelRepresentation(script_module)


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    input_path = sys.argv[2]

    torch.set_grad_enabled(False)

    rave_model = load_model(model_path)

    input_sample = read_audio_sample(input_path)
    # TODO: if the sampling rate is different, resample
    fs = input_sample.sampling_rate
    if fs != rave_model.sampling_rate:
        print(f"sampling rates differ; input: {fs}, rave: {rave_model.sampling_rate}")

    input_size = int(2*fs)
    data = input_sample.data[:input_size]
    normalized_data = data / np.max(np.abs(data), axis=0)
    frames = torch.Tensor(normalized_data)
    input_ = torch.reshape(frames, (1, 1, input_size))

    result = rave_model.forward(input_)
    play(result[0][0], fs)
