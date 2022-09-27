# flower_classification_model
Computer vision model for scivision, classifying flower species

## Installation

To install this package into your environment, run `pip install .` from this directory.
*Note for MacOS users with new M1/M2 machines*: Tensorflow doesn't seem to be installable from pip for ARM-native python.  Can either use Intel-compiled python (e.g. via anaconda), or install tensorflow via conda.  If the latter, you may need to remove tensorflow from `requirements.txt` to avoid problems when running `load_pretrained_model` in Scivision.


## Contents of this package

The python code itself is in the directory `honeybee_species_model/`
* `__init__.py` just allows this directory to be treated as a python package, and imports the contents of `model.py` into the package's namespace.
* `model.py` this is the wrapper for the model itself.   It ensures that images on which the model will be run are the correct size, and it provides the interface that Scivision expects.   It should contain a single class, which has a `predict` method.   It should also have a way of loading the weights.
