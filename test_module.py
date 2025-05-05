import mlp


model = mlp.mlp([
    mlp.mlp_layers.mlp_layer(5, activation="sigmoid", weight_initializer="uniform"),
    mlp.mlp_layers.mlp_layer(3, activation="sigmoid", weight_initializer="normal"),
    mlp.mlp_layers.mlp_layer(5, activation="sigmoid", weight_initializer="uniform")
])

print("\n TESTING INSTANCIATION FROM CSV !")
model2 = mlp.mlp(source="data.csv")

print("\n TESTING PRINTING METHOD !")
model.print_layers(1)

model.import_dataset()
