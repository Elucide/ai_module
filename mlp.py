import numpy as np
import mlp_layers

class mlp:
    """
        MULTILAYER PERCEPTRON SUPER COOL CLASS !!!
    """

    def __init__(self, nn_layers: list[mlp_layers.mlp_layer]):
        self.layers = nn_layers
        print("Neural network created !")
        for layer in self.layers:
            print(layer)
    
    def print_layers(self, n : int = 0):
        if n:
            print (str(self.layers[n]))
        else:
            for layer in self.layers:
                print (str(layer))

        
