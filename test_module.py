import mlp

model = mlp.mlp([
    mlp.mlp_layers.mlp_layer(5, activation="sigmoid", weight_initializer="uniform"),
    mlp.mlp_layers.mlp_layer(5, activation="sigmoid", weight_initializer="uniform"),
    mlp.mlp_layers.mlp_layer(5, activation="sigmoid", weight_initializer="uniform")
])

print("model content:", model.layers[0][1])
