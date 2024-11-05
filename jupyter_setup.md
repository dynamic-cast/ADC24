# Setting up the Playground
Get comfty on your command line for this set up. You might be able to run the following lines from your IDE directly if it renders .md files and supports runing bash prompts from them. You can use your IDE terminal, Terminal on MacOs or Linux, or Command Prompt on Windows.

## Get our code from github
Use your favourite git client to clone our workshop [repo](https://github.com/dynamic-cast/ADC24).
This will include the example app and complete workshop.
```bash
git clone git@github.com:dynamic-cast/ADC24.git
```

For the first part of the workshop, please check out our branch
```bash
git checkout workshop/part-1-embed-model
```

## Setting up a local environment
For this project, you'll need a python version between 3.8. and 3.12 (numpy does not support 3.13) Check your version with:

```bash
python --version
```

please follow [this guide](https://www.pythoncentral.io/how-to-update-python/) if you need to update your python version.

We don't want to mess with your favourite personal python set up, so we'll create a local environment and activate it. If you have python 3.13 installed, please specify the version below by replacing `python3` with e.g. `python3.12`. On Mac you can run: 

```bash
python3 -m venv workshopenv
```
```bash
source workshopenv/bin/activate
```

On Windows:
```bash
python3 -m venv workshopenv
```
```bash
workshopenv\Scripts\activate.bat
```

All the modules we need are defined in requirement.txt. You can install them like this after you activated your local environment
```bash
python3 -m pip install -r requirements.txt
```

If pip has problems finding the right packages, try to upgrade pip
```bash
python3 -m pip install --upgrade pip
```

## Prepare environment for jupyter notebook 

Now we have our environment with all modules. We need to run a couple more commands so we can select this later in jupyter notebook.
```bash
python3 -m ipykernel install --user --name=workshopenv
```

## Run jupyter notebook
Only one more thing to do on the command line to start up jupyter notebook. 
```bash
jupyter notebook
```

This will make a browser pop up showing a file index. Open playground.ipynb.
If the browser does not pop up automatically, you can click on the link in the command line output:
![Link to local jupyter host](jupyter_setup_resources/jupyter_server.jpg)

## Select Local Environment
To select our workshop environment in jupyter notebook you can follow these steps:
1. Select the Kernel dropdown
![Select Kernel menu](jupyter_setup_resources/select_kernel1.jpg)
2. Click change Kernel and choose the new Kernel
![Change Kernel](jupyter_setup_resources/select_kernel2.jpg)
3. Select it
![Select Kernel](jupyter_setup_resources/select_kernel3.jpg)

## Downloading the RAVE models
You can use the line from the notebook to download the model, if you're on mac and haven't downloaded files from python before you might need to follow the step below.
If you prefer you can download the models manually [here](https://acids-ircam.github.io/rave_models_download) and pass the path to the file when creating the audio engine.   

### Mac Users
#### Install certificates so you can download the model file
Use the Spotlight search to find the file that executes the command. You can press command + space bar to open the search bar and start typing "Install Certificate.command"
![Install the certificate](jupyter_setup_resources/install_certificates.jpg)
You can double click on the entry, this will open a terminal and install the certificate needed to verify HTTPS connections.

Now you can execute the lines in the notebook and make some noise, yay.