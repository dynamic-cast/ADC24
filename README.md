# Style Transfer Demo App

This app loads a [RAVE](https://github.com/acids-ircam/RAVE) model and enables navigating its latent space with controls.

This is a Python app with a web UI. In order to run it:
* Download one of the RAVE models from [here](https://acids-ircam.github.io/rave_models_download).
This app is tested with Darbouka_onnx
* Set the path to the model in `instance/application.cfg`
* Install the dependencies:
`pip install -r requirements.txt`
* Run the app:
`flask run`

In order to run the app in debug mode, run:
`flask run --debug`
