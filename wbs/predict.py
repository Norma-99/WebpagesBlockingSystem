import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from wbs.preprocessing import SamplePreprocessing

class SamplePredict:
    def __init__(self, url:str, model_path):
        self.url = url
        self.df = SamplePreprocessing(self.url).preprocess()
        self.model = keras.models.load_model(model_path)
        self.model.summary()

    def predict(self):
        self.predictions = self.model.predict(self.df)
        print(self.predictions[0])
        return self.predictions[0]

class ValidationPredict:

    def __init__(self, val_dataset_path, model_path):

        def load_data(path:str):
            with open(path, 'rb') as f:
                return pickle.load(f)
        
        self.x_val, self.y_val = load_data(val_dataset_path)
        self.model = keras.models.load_model(model_path)
        self.model.summary()

    def predict(self):
        self.x_val = self.x_val.to_numpy()
        print(self.x_val[0])
        self.predictions = self.model.predict(self.x_val)
    
    def analyze(self):
        self.predictions = np.reshape(self.predictions, self.predictions.shape[0])
        self.predictions[self.predictions < 0.5] = 0.0
        self.predictions[self.predictions > 0.5] = 1.0

        self.y_val = self.y_val.to_numpy()
        self.y_val = np.reshape(self.y_val, self.y_val.shape[0])

        err = np.sum(np.abs(self.predictions - self.y_val))

        print("Number of erroneous predictions: ", err)
        print("Pertentage of erroneous predictions: ", (err * 100) / self.predictions.shape[0], "%")
        # Percentage of malicious websites
        malicious = 0
        for i in self.y_val:
            if i == 1:
                malicious += 1

        print("Number of malicious websites: ", malicious)
        print("Percentage of malicious websites: ", (malicious*100)/self.predictions.shape[0], "%")

        # Number of malicious websites wrongly classified (y_val = 1 & pred = 0)
        mal_err = 0
        for i, j in zip(self.y_val, self.predictions):
            if (i == 1.0) & (j == 0.0) :
                mal_err += 1

        print("Number of wrongly classified malicious websites: ", mal_err)
        print("Percentage of wrongly classified malicious websites: ", (mal_err*100)/malicious, "%")
    
    def run(self):
        self.predict()
        self.analyze()

# ValidationPredict("../Datasets/val_dataset.pickle", "../Results/model.h5").run()
#SamplePredict("http://www.domain.com", "../Results/model.h5").predict()