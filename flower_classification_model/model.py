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

CLASS_LABELS = [
    "-1",
    "Mixed local stock",
    "Carniolan honey bee",
    "Italian honey bee",
    "Russian honey bee",
    "VSH Italian honey bee",
    "Western honey bee"
]

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
        image = resize(image, (70, 70),
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
