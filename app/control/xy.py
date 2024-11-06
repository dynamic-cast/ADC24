import copy
import numpy as np
import torch
import torch.nn as nn
import json

class Model(nn.Module):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

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
    def __init__(self, set_controls_callback, send_msg_callback, model, *a, **kw):
        super().__init__(*a, **kw)

        self._set_controls_callback = set_controls_callback
        self._send_msg_callback = send_msg_callback
        self._mode = Mode.none
        self._training_data = TrainingData()
        self._model_weights = None
        self._model = model
        self._model_trained = False

    def log(self, message):
        # logger.info(message)
        if self._send_msg_callback:
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
            self.log(f"Passing data: {mouse_xy}")
            self.generate_latent_coordinates(mouse_xy)

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
                self.load_data()  # Load from file before training
            if self._training_data.empty:
                self.log("No training data available")
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
        X_train, X_test, y_train, y_test = self.prepare_data()

        # Initialize model, loss function, and optimizer
        self.initialize_training_components()

        # Hold the best model
        best_mse = np.inf # init to infinity
        history = []

        # Training loop
        for epoch in range(self.n_epochs): # n_epochs = 100
            running_loss = self.train_one_epoch(X_train, y_train)

            # evaluate accuracy at end of each epoch
            mse = self.validate(X_test, y_test)
            history.append(mse)

            # Save best model if improved
            if mse < best_mse:
                best_mse = mse
                self._model_weights = copy.deepcopy(self._model.state_dict())

            self.log(f"Epoch [{epoch+1}/{self.n_epochs}], Loss: {running_loss:.4f}, MSE: {mse:.4f}")

        self.log("Training finished")
        self._model_trained = True

    def load_data(self, file_path="training_data.json"):
        """
        Load data from a file and populate training data.
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        # Clear current training data
        self._training_data.reset()

        for point in data:
            mouse_xy = point["mouse_xy"]
            latent_coordinates = point["latent_coordinates"]
            self._training_data.add_data_point(mouse_xy, latent_coordinates)

        self.log("Training data loaded from file")

    def prepare_data(self):
        X = torch.tensor(self._training_data.training_inputs, dtype=torch.float32)
        y = torch.tensor(self._training_data.training_outputs, dtype=torch.float32)

        num_samples = X.size(0)
        indices = torch.randperm(num_samples)

        split_idx = int(num_samples * 0.8)
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        return X_train, X_test, y_train, y_test

    
    def initialize_training_components(self):
        self.loss_fn = nn.MSELoss()
        self.optimiser = torch.optim.Adam(self._model.parameters(), lr=0.0001)
        self.n_epochs = 100
        self.batch_size = 5

    def train_one_epoch(self, X_train, y_train):
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
            if not loss.requires_grad:
                loss.requires_grad_(True)
                
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            running_loss += loss.item()
        
        return running_loss / len(batch_start)  # Average loss per batch
    
    def validate(self, X_test, y_test):
        self._model.eval()
        with torch.no_grad():
            y_pred = self._model(X_test)
            mse = self.loss_fn(y_pred, y_test).item()
        return mse
    
    def prepare_control(self):
        self._model.load_state_dict(self._model_weights)
        self._model.eval()

    def generate_latent_coordinates(self, mouse_xy):
        with torch.no_grad():
            x = torch.tensor(mouse_xy, dtype=torch.float32)
            y_pred = self._model(x)
            self._set_controls_callback(y_pred)
            if self._send_msg_callback:
                self._send_msg_callback("set_dims", {"dims": [float(v) for v in y_pred]})
