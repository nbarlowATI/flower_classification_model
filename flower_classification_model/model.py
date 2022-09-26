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


class efficientNetB3:
    def __init__(self):
        filename = "flowers_98.04.h5"
        if not os.path.exists(filename):
            model_path = os.path.join("https://connectionsworkshop.blob.core.windows.net/flowers", filename)
            r = requests.get(model_path)
            with open(filename, "wb") as outfile:
                outfile.write(r.content)
        self.model = tf.keras.models.load_model(filename)


    def predict(self, image: np.ndarray):
        ### resize all images to the size expected by the network
        image = resize(image, (200, 244),
                   preserve_range=True,
                   anti_aliasing=True)
        image = np.expand_dims(image, 0)
        result = self.model.predict(image)
        # find the highest weight, and corresponding class name and index
        max_index = 0
        max_result = ""
        max_weight = 0.
        for i, weight in enumerate(result[0]):
            if weight > max_weight:
                max_weight = weight
                max_result = CLASS_LABELS[i]
                max_index = i
        return "{}: {:.2f}%".format(max_result, max_weight*100)
