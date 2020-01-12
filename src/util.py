from neural_network import init_nn

model = init_nn(8, 4, [16,12])

input_layer = [0, -0.23239786] + [0] * 6

print(input_layer)

print(model.predict(input_layer))