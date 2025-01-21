import numpy as np
import mlp_layers

class mlp:
    """
        MULTILAYER PERCEPTRON SUPER COOL CLASS !!!
    """

    
    def print_layers(self, n : int = 0):
        if n:
            print (str(self.layers[n]))
        else:
            for layer in self.layers:
                print (str(layer))

    def get_model_from_csv(self, source):
        print ("Time to get that model from", source, "...")
        return ({})

        
    def __init__(self, nn_layers: list[mlp_layers.mlp_layer]=None, source=None):
        if (source != None):
            self.layers = self.get_model_from_csv(source)
        elif(nn_layers != None):
            self.layers = nn_layers
        else:
            raise ValueError(f"wrong arguments passed to mlp constructor:\n\tat least one way to initialize a model should be provided !")
        print("Neural network created !")
        for layer in self.layers:
            print(layer)
