import numpy as np
import mlp_layers

class mlp:
    """
        MULTILAYER PERCEPTRON SUPER COOL CLASS !!!
    """

    def __init__(self, nn_layers: list[mlp_layers.mlp_layer]):
        self.layers = nn_layers
        print("Neural network created !")
        print("it's layers are:", self.layers)

        
