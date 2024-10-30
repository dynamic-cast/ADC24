from sklearn.model_selection import train_test_split

import copy
import logging
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

    @property
    def empty(self):
        return len(self._training_inputs) == 0

class XYControl:
    def __init__(self, set_controls_callback, send_msg_callback, *a, **kw):
        super().__init__(*a, **kw)

        self._set_controls_callback = set_controls_callback
        self._send_msg_callback = send_msg_callback
        self._mode = Mode.none
        self._training_data = TrainingData()
        self._model_weights = None
        self._model = Model()
        self._model_trained = False

    def log(self, message):
        # logger.info(message)
        self._send_msg_callback("log", message)

    def set_mode(self, mode):
        self._mode = mode
        if mode == Mode.data_gathering:
            self._training_data.reset()

    def receive_coordinates(self, mouse_xy, latent_coordinates):
        if self._mode == Mode.data_gathering:
            self.log(f"Adding data point: {mouse_xy} -> {latent_coordinates}")
            self._training_data.add_data_point(mouse_xy, latent_coordinates)
        if self._mode == Mode.control and self._model_trained:
            self.log(f"Passing data: {mouse_xy} -> {latent_coordinates}")
            self.generate_latent_coordinates(mouse_xy, latent_coordinates)

    def _toggle_mode(self, active_mode, value):
         if value == "on":
             self.set_mode(active_mode)
         elif value == "off" and self._mode == active_mode:
             self.set_mode(Mode.none)

    def toggle_data_gathering(self, value):
        self._toggle_mode(Mode.data_gathering, value)

    def toggle_training(self, value):
        self._toggle_mode(Mode.training, value)
        if value == "on":
            if self._training_data.empty:
                self.log("No training data")
                return
            self.train()

    def toggle_control(self, value):
        self._toggle_mode(Mode.control, value)
        if value == "on":
            if self._model_trained:
                self.prepare_control()
            else:
                self.log("Model not trained yet")

    def train(self):
        self.log("Training started")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self._prepare_data()
        
        # Initialize model, loss function, and optimizer
        self._initialize_training_components()
        
        # Hold the best model
        best_mse = np.inf # init to infinity
        history = []

        # Training loop
        for epoch in range(self.n_epochs): # n_epochs = 100
            running_loss = self._train_one_epoch(X_train, y_train)
            
            # evaluate accuracy at end of each epoch
            mse = self._validate(X_test, y_test)
            history.append(mse)
            
            # Save best model if improved
            if mse < best_mse:
                best_mse = mse
                self._model_weights = copy.deepcopy(self._model.state_dict())

            self.log(f"Epoch [{epoch+1}/{self.n_epochs}], Loss: {running_loss:.4f}, MSE: {mse:.4f}")

        self.log("Training finished")
        self._model_trained = True

    def _prepare_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self._training_data.training_inputs,
            self._training_data.training_outputs,
            train_size=0.8,
            shuffle=True
        )
        # Convert to PyTorch tensors
        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
    
    def _initialize_training_components(self):
        self._model = Model()
        self.loss_fn = nn.MSELoss()
        self.optimiser = torch.optim.Adam(self._model.parameters(), lr=0.0001)
        self.n_epochs = 100
        self.batch_size = 5

    def _train_one_epoch(self, X_train, y_train):
        running_loss = 0.0
        batch_start = torch.arange(0, len(X_train), self.batch_size)
        self._model.train()

        for start in batch_start:
            X_batch = X_train[start:start+self.batch_size]
            y_batch = y_train[start:start+self.batch_size]

            # Forward pass
            y_pred = self._model(X_batch)
            loss = self.loss_fn(y_pred, y_batch)
            
            # Backward pass and optimization
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            running_loss += loss.item()
        
        return running_loss / len(batch_start)  # Average loss per batch
    
    def _validate(self, X_test, y_test):
        self._model.eval()
        with torch.no_grad():
            y_pred = self._model(X_test)
            mse = self.loss_fn(y_pred, y_test).item()
        return mse
    
    def prepare_control(self):
        self._model.load_state_dict(self._model_weights)
        self._model.eval()

    def generate_latent_coordinates(self, mouse_xy, latent_coordinates):
        with torch.no_grad():
            x = torch.tensor(mouse_xy, dtype=torch.float32)
            y_pred = self._model(x)
            self._set_controls_callback(y_pred)
            self._send_msg_callback("set_dims", {"dims": [float(v) for v in y_pred]})
