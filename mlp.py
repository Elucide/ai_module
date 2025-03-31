import numpy as np
import mlp_layers

class mlp:
    """
        MULTILAYER PERCEPTRON SUPER COOL CLASS !!!

        can be instatianted with :
            "mlp my_new_mlp(<nn_layers>, <source>)"
    
                (-) <nn_layers> a list containing the layers of the neural network
                (-) <source> a string containing a path to the csv used to instanciate the layers

            only one argument should be passed to the object constructor
    """
    
    def monitor(self, n : int = 0):
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


    def print_layers(self, index = None):
        if (index == None):
            for layers in self.layers:
                layers.monitor()
        else:
            self.layers[index].monitor()
