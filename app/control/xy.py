from sklearn.model_selection import train_test_split

import copy
import numpy as np
import torch
import torch.nn as nn
import tqdm

class Model(nn.Module):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        # pyramid
        # self.layers = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 4),
        # )

    def forward(self, x):
        """
        Forward pass
        """
        return self.layers(x)

class Mode:
    data_gathering = 0
    training = 1
    control = 2
    none = 3

class TrainingData:
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self._training_inputs = []
        self._training_outputs = []

    def add_data_point(self, x, y):
        self._training_inputs.append(x)
        self._training_outputs.append(y)

    def reset(self):
        self._training_inputs = []
        self._training_outputs = []

    @property
    def training_inputs(self):
        return self._training_inputs

    @property
    def training_outputs(self):
        return self._training_outputs

class XYControl:
    def __init__(self, set_controls_callback, *a, **kw):
        super().__init__(*a, **kw)

        self._set_controls_callback = set_controls_callback
        self._mode = Mode.none
        self._training_data = TrainingData()
        self._model_weights = None
        self._model = Model()

        self._new_dimensions_ready = False

    @property
    def new_dimensions_ready(self):
        return self._new_dimensions_ready

    @new_dimensions_ready.setter
    def new_dimensions_ready(self, ready):
        self._new_dimensions_ready = ready

    def set_mode(self, mode):
        self._mode = mode
        if mode == Mode.data_gathering:
            self._training_data.reset()

    def receive_coordinates(self, mouse_xy, latent_coordinates):
        if self._mode == Mode.data_gathering:
            print(f"Adding data: {mouse_xy}, {latent_coordinates}")
            self._training_data.add_data_point(mouse_xy, latent_coordinates)
        if self._mode == Mode.control:
            print(f"Passing data: {mouse_xy}, {latent_coordinates}")
            self.generate_latent_coordinates(mouse_xy, latent_coordinates)

    def _toggle_mode(self, active_mode, value):
        if value == "on":
            self.set_mode(active_mode)
        elif value == "off":
            self.set_mode(Mode.none)

    def toggle_data_gathering(self, value):
        self._toggle_mode(Mode.data_gathering, value)

    def toggle_training(self, value):
        self._toggle_mode(Mode.training, value)
        if value == "on":
            self.train()

    def toggle_control(self, value):
        print(f"toggle control, {value}")
        self._toggle_mode(Mode.control, value)
        if value == "on":
            self.prepare_control()

    def train(self):
        print("Training started")
        X_train, X_test, y_train, y_test = train_test_split(
            self._training_data.training_inputs,
            self._training_data.training_outputs,
            train_size=0.8,
            shuffle=True)
        len_data = len(X_train)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        self._model = Model()
        loss_fn = nn.MSELoss()
        optimiser = torch.optim.Adam(self._model.parameters(), lr=0.0001)

        n_epochs = 100
        batch_size = 5
        batch_start = torch.arange(0, len(X_train), batch_size)

        # Hold the best model
        best_mse = np.inf # init to infinity
        history = []

        for epoch in range(n_epochs):
            running_loss = 0.0
            self._model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train[start:start+batch_size]
                    y_batch = y_train[start:start+batch_size]
                    # forward pass
                    y_pred = self._model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    # backward pass
                    optimiser.zero_grad()
                    loss.backward()
                    # update weights
                    optimiser.step()
                    # print progress
                    bar.set_postfix(mse=float(loss))
                    running_loss += loss.item()

            # evaluate accuracy at end of each epoch
            self._model.eval()
            y_pred = self._model(X_test)
            mse = loss_fn(y_pred, y_test)
            mse = float(mse)
            history.append(mse)
            if mse < best_mse:
                best_mse = mse
                self._model_weights = copy.deepcopy(self._model.state_dict())
            print(f"Epoch [{epoch+1}/{n_epochs}], " +
                  f"Loss: {running_loss/len_data:.4f}")

        print("Training finished")

    def prepare_control(self):
        print(f"prepare control")
        self._model.load_state_dict(self._model_weights)
        self._model.eval()

    def generate_latent_coordinates(self, mouse_xy, latent_coordinates):
        with torch.no_grad():
            x = torch.tensor(mouse_xy, dtype=torch.float32)
            y_pred = self._model(x)
            print(f"current coordinates: {latent_coordinates}")
            print(f"new coordinates: {y_pred}")
            self._set_controls_callback(y_pred)
            self._new_dimensions_ready = True
