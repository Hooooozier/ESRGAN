# ESRGAN
ESRGAN implemented by torch, used for paper experiment

## Model
Implement SRGAN/ESRGAN based on BasicSR repo, just removed some redundant file to avoid heavy loadup.

## Usage
Specify the train and validation dataset path in the config.py, some training parameters are also here.
The dataset should only provides HR images in the folder. The dataset.py will generate LR/HR training pairs automatically with the help of PIL.

## environment setup
```shell
pip install -r requirements.txt
```
