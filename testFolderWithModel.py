import os
import numpy as np
from tacotron2.layers import TacotronSTFT
import torch
from scipy.io.wavfile import read
from LinearRegressionModel import LinearRegressionModel

class ComparatorClass():
    def __init__(self,testFolder,modelPath):
        self.wav_file_list = os.listdir(testFolder)
        self.model_input_size = 4
        self.model_output_size = 5
        self.first_hidden_layer_factor = 5
        self.second_hidden_layer_factor = 5
        self.model = LinearRegressionModel(self.model_input_size, self.model_output_size,
                                        self.first_hidden_layer_factor, self.second_hidden_layer_factor)
        self.model.load_state_dict(torch.load(modelPath))

    def processTesting(self):
        for (i,file) in enumerate(self.wav_file_list):
            print("....")
            #convert audio to stft, mel
            #rewrite converter function so that they can be used from here.
            #convert predict audio from that mel
            #convert that back to stft, mels



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-tf', "--testFolder", dest='testFolder')
    parser.add_argument('-m', "--modelPath", dest='modelPath')

    args = parser.parse_args()

    comparator = ComparatorClass(args.testFolder,args.modelPath)

