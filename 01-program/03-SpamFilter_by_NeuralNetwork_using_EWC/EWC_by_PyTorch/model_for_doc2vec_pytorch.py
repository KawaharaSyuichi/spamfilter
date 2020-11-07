import torch
import torch.nn as nn

IN_DIM = 300
MIDDLE_DIM = 100
OUT_DIM = 2


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.input_layer = nn.Linear(IN_DIM, MIDDLE_DIM)
        self.output_layer = nn.Linear(MIDDLE_DIM, OUT_DIM)

        nn.init.normal_(self.input_layer.weight, 0.0, 1.0)
        self.input_layer.bias.data.fill_(0.1)
        nn.init.normal_(self.output_layer.weight, 0.0, 1.0)
        self.output_layer.bias.data.fill_(0.1)

    def forward(self, input):
        x = self.input_layer(input)
        x = self.output_layer(x)
        x = self.relu(x)

        return x


model = NeuralNetwork()

print(model)
for param in model.parameters():
    print(param)
