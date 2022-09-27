import os
import requests
import numpy as np
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
# prevent annoying tensorflow warning

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import warnings
warnings.simplefilter("ignore")

CLASS_LABELS = ['alpine sea holly', 'anthurium', 'artichoke', 'azalea', 'ball moss', 'balloon flower', 'barbeton daisy', 'bearded iris', 'bee balm', 'bird of paradise', 'bishop of llandaff', 'black-eyed susan', 'blackberry lily', 'blanket flower', 'bolero deep blue', 'bougainvillea', 'bromelia', 'buttercup', 'californian poppy', 'camellia', 'canna lily', 'canterbury bells', 'cape flower', 'carnation', 'cautleya spicata', 'clematis', "colt's foot", 'columbine', 'common dandelion', 'corn poppy', 'cyclamen', 'daffodil', 'desert-rose', 'english marigold', 'fire lily', 'foxglove', 'frangipani', 'fritillary', 'garden phlox', 'gaura', 'gazania', 'geranium', 'giant white arum lily', 'globe thistle', 'globe-flower', 'grape hyacinth', 'great masterwort', 'hard-leaved pocket orchid', 'hibiscus', 'hippeastrum', 'japanese anemone', 'king protea', 'lenten rose', 'lotus', 'love in the mist', 'magnolia', 'mallow', 'marigold', 'mexican aster', 'mexican petunia', 'monkshood', 'moon orchid', 'morning glory', 'orange dahlia', 'osteospermum', 'oxeye daisy', 'passion flower', 'pelargonium', 'peruvian lily', 'petunia', 'pincushion flower', 'pink primrose', 'pink-yellow dahlia?', 'poinsettia', 'primula', 'prince of wales feathers', 'purple coneflower', 'red ginger', 'rose', 'ruby-lipped cattleya', 'siam tulip', 'silverbush', 'snapdragon', 'spear thistle', 'spring crocus', 'stemless gentian', 'sunflower', 'sweet pea', 'sweet william', 'sword lily', 'thorn apple', 'tiger lily', 'toad lily', 'tree mallow', 'tree poppy', 'trumpet creeper', 'wallflower', 'water lily', 'watercress', 'wild pansy', 'windflower', 'yellow iris']

def load_model_with_weights(url):
    ### TODO ####
    ### Need to download the weights file from url, if it's not already
    ### present, and put the downloaded filename into a variable
    ### called weights_filename
    model = tf.keras.models.load_model(weights_filename)
    return model

def preprocess_image(image):
    """
    Ensure that an input image is the correct size, and
    has the expected shape, to be used by the predict function

    parameters
    ----------
    image: np.ndarray, shape(npix_x,npix_y,3)

    returns
    -------
    image: np.ndarray, shape(None, 200, 244, 3)
    """
    image = resize(image, (200, 244),
                   preserve_range=True,
                   anti_aliasing=True)
    image = np.expand_dims(image, 0)
    return image


class efficientNetB3:
    ### TODO ####
    ### Add a constructor to this class that calls the function
    ### to download the model weights, load the model, and assign
    ### to self.model

    def predict(self, image: np.ndarray):
        ### TODO - make sure the image is the correct size, and has
        ### the dimensions expected by the model.

        result = self.model.predict(image)
        ### TODO ####
        ### Find the highest weight, and, using the list of CLASS_LABELS
        ### get the corresponding class name.
        return "FIXME"



if __name__ == "__main__":
    pass
