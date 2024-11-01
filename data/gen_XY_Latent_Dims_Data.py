import numpy as np
import json

def generate_synthetic_data(num_samples=100):
    mouse_coordinates = np.random.uniform(-1, 1, (num_samples, 2))
    latent_coordinates = np.random.uniform(-1, 1, (num_samples, 4))

    data = [
        {"mouse_xy": mouse_coordinates[i].tolist(), "latent_coordinates": latent_coordinates[i].tolist()}
        for i in range(num_samples)
    ]

    with open("training_data.json", "w") as f:
        json.dump(data, f)

def generate_clamped_synthetic_data(num_samples=100):
    x_values = np.random.uniform(1000, 1500, num_samples)
    y_values = np.random.uniform(100, 650, num_samples)
    mouse_coordinates = np.column_stack((x_values, y_values))

    latent_coordinates = np.random.uniform(-1, 1, (num_samples, 4))

    data = [
        {"mouse_xy": mouse_coordinates[i].tolist(), "latent_coordinates": latent_coordinates[i].tolist()}
        for i in range(num_samples)
    ]

    with open("clamped_training_data.json", "w") as f:
        json.dump(data, f)

generate_synthetic_data()
generate_clamped_synthetic_data()
