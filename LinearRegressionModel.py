import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim,first_hidden_layer_factor,second_hidden_layer_factor):
        super(LinearRegressionModel, self).__init__()
        print("input_dim", input_dim)
        print("output_dim", output_dim)
        hidden_layer_size = input_dim * first_hidden_layer_factor
        second_hidden_layer_size = input_dim * second_hidden_layer_factor

        self.sequencial =nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, second_hidden_layer_size),
            nn.ReLU()
        )
        self.linearImag = nn.Sequential(
            nn.Linear(second_hidden_layer_size, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim)
        )

        self.linearReal = nn.Sequential(
            nn.Linear(second_hidden_layer_size, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim)
        )
        self.linearMag = nn.Sequential(
            nn.Linear(second_hidden_layer_size, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim)
        )

    def forward(self, x):
        intermediate = self.sequencial(x)
        imag = self.linearImag(intermediate)
        real = self.linearReal(intermediate)
        mag = self.linearMag(intermediate)
        #
        return imag,real,mag