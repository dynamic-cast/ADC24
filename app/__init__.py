import os

from flask import Flask, flash, g, render_template
from app.audio import create_audio_engine


def get_audio_engine(input_wav):
    if 'audio_engine' not in g:
        g.audio_engine = create_audio_engine(input_wav)

    return g.audio_engine


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

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
        audio_engine = get_audio_engine(app.config['INPUT_WAV'])
        audio_engine.start()

        return render_template("main.html")

    return app

    @app.teardown_appcontext
    def teardown_audio_engine(exception):
        audio_engine = g.pop('audio_engine', None)

        if audio_engine is not None:
            audio_engine.stop.set()
