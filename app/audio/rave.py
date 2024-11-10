import numpy as np
import torch

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
