from flask import Flask, current_app, g, render_template
from flask_socketio import SocketIO, emit

from .app_utils import reset_audio_engine, setup_app

app = Flask(__name__, instance_relative_config=True)
socketio = SocketIO(app)
setup_app(app, socketio)

@app.route('/')
def index():
    reset_audio_engine(current_app.audio_engine)
    return render_template("main.html")

@socketio.on("dims")
def handle_dims(dim1, dim2, dim3, dim4):
    coordinates = [float(dim1), float(dim2), float(dim3), float(dim4)]
    current_app.audio_engine.set_latent_coordinates(coordinates)

def to_floats(arr):
    return [float(v) for v in arr]

@socketio.on("add_data_point")
def handle_add_data_point(json):
    xy = to_floats(json["xy"])
    dims = to_floats(json["dims"])
    current_app.xy_control.receive_coordinates(xy, dims)

def toggle(event_obj, val):
    if val == "on":
        event_obj.set()
    elif val == "off":
        event_obj.clear()

@socketio.on("toggle")
def handle_toggle(json):
    name = json["element"]
    val = json["state"]
    if name == "playing":
        toggle(current_app.audio_engine.stop, "on" if val == "off" else "off")
    elif name == "looping":
        toggle(current_app.audio_engine.loop, val)
    elif name == "transforming":
        toggle(current_app.audio_engine.transform, val)
    elif name == "gathering":
        current_app.xy_control.toggle_data_gathering(val)
    elif name == "training":
        current_app.xy_control.toggle_training(val)
    elif name == "controlling":
        current_app.xy_control.toggle_control(val)

if __name__ == '__main__':
    socketio.run(app)

if __name__ == '__main__':
    socketio.run(app)
