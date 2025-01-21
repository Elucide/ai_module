import numpy as np


class mlp_layer:
    """
        MLP LAYER SUPER COOL CLASS
    """

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x):
        return np.tanh(x)

    def _linear(self, x):
        return x

    def _forward(self, x):
        return self.activate(x)

    def _uniform_init(self, shape, low=0, high=1):
        return np.random.uniform(low, high, shape)

    def _normal_init(self, shape, mean=0, std=1):
        return np.random.normal(mean, std, shape)

    def __init__(self, size, activation="sigmoid", weight_initializer="normal", bias_initializer=0.0):
        print ("super cool mlp layer class created")
        self.weights = np.ndarray(size)
        self.bias = np.float64(bias_initializer)

        #self.activation_name = activation
        self.activation_functions = {
            "relu": self._relu,
            "sigmoid": self._sigmoid,
            "tanh": self._tanh,
            "linear": self._linear,
            "forward": self._forward
        }

        if activation not in self.activation_functions:
            raise ValueError(f"Activation function is not supported !")
        self.activate = self.activation_functions[activation]

        self.weight_initializers = {
            "uniform": self._uniform_init,
            "normal": self._normal_init
        }

        if weight_initializer not in self.weight_initializers:
            raise ValueError(f"initializer not supported !")
        self.weight_initializer = self.weight_initializers[weight_initializer]

        #init the weights !
        self.weights = self.weight_initializer(len(self.weights))


    def __getitem__(self, index):
        return self.weights[index]

    def __setitem__(self, index, value):
        self.weights[index] = value

    def __str__(self):
        out_str = "weights are: "+ str(self.weights)+ "\nbias are: "+ str(self.bias)
        return (out_str)
