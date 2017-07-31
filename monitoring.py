import storage
import keras
from flask_jsonpify import json


class TrainingLogs(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
