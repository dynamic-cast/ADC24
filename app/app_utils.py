from app.audio import create_audio_engine
from app.control.xy import XYControl, Model

import os

def setup_app(app, socketio):
    app.config.from_pyfile('application.cfg', silent=True)
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

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

    app.xy_control = XYControl(app.audio_engine.set_latent_coordinates, socketio.emit, Model())

def reset_audio_engine(audio_engine):
    audio_engine.stop.set()
    audio_engine.loop.clear()
    audio_engine.transform.clear()
