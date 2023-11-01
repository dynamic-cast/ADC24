import os

from flask import Flask, current_app, flash, g, render_template
from flask_sock import Sock
from queue import Queue

sock = Sock()
from app.audio import create_audio_engine

def get_audio_engine(input_wav, model):
    if 'audio_engine' not in g:
        g.audio_engine = create_audio_engine(input_wav, model)

    return g.audio_engine

def toggle(event_obj, val):
    if val == "on":
        event_obj.set()
    elif val == "off":
        event_obj.clear()

@sock.route('/dims')
def latent_dimensions(ws):
    while True:
        data = ws.receive()
        coordinates = [float(val) for val in data.split(",")]
        current_app.audio_engine.set_latent_coordinates(coordinates)

@sock.route('/toggle')
def toggle_switches(ws):
    while True:
        data = ws.receive()
        name, val = [elem.strip() for elem in data.split(",")]
        if name == "playing":
            toggle(current_app.audio_engine.stop, "on" if val == "off" else "off")
        elif name == "looping":
            toggle(current_app.audio_engine.loop, val)
        elif name == "transform":
            toggle(current_app.audio_engine.transform, val)

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    sock.init_app(app)

    if test_config is None:
        app.config.from_pyfile('application.cfg', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def main():
        audio_engine.stop.set()
        audio_engine.loop.clear()
        audio_engine.transform.clear()
        return render_template("main.html")

    # BODGE!
    # app.latent_coordinates = Queue(maxsize=1)
    audio_engine = create_audio_engine(
        app.config['INPUT_WAV'],
        app.config['MODEL'],
        # app.latent_coordinates,
    )
    audio_engine.stop.set()
    app.audio_engine = audio_engine
    audio_engine.start()

    return app
