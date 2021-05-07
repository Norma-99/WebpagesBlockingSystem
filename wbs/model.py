import pandas as pd
import numpy as np
import pickle
import json
import os

import tensorflow as tf 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Precision, Recall

class Model:
    def __init__(self, train_path:str, test_path:str, config_file):

        def load_data(path:str):
            with open(path, 'rb') as f:
                return pickle.load(f)

        def read_config(path:str):
            with open(path) as conf_file:
                return json.load(conf_file)
        
        self.x_train, self.y_train = load_data(train_path)
        self.x_test, self.y_test = load_data(test_path)
        self.config = read_config(config_file)

    def create_model(self):
        self.model = Sequential([
        Dense(self.config['neurons_l1'], activation='relu', input_shape=(45,)),
        Dense(self.config['neurons_l2'], activation='relu'),
        Dense(self.config['neurons_l3'], activation='relu'),
        Dense(1, activation='sigmoid')])

        self.model.summary()
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', 
                    metrics=['accuracy', AUC(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives(), Precision(), Recall()]) 

    def train_model(self, results_path:str):
        history = self.model.fit(self.x_train, self.y_train, epochs=self.config['epochs'], validation_data=(self.x_test, self.y_test), verbose=2)
        # convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history) 

        # Save to csv: 
        with open(results_path, mode='w') as f:
            hist_df.to_csv(f)

    def export_model(self, model_path:str):
        self.model.save(model_path)

    def run(self, results_path:str, model_path:str):
        self.create_model()
        self.train_model(results_path)
        self.export_model(model_path)


# Model('../Datasets/train_dataset.pickle', '../Datasets/test_dataset.pickle', "../Configs/config.json").run("../Results/NN.csv", "../Results/model.h5")