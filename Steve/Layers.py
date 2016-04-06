class Layers:
    def __init__(self, num_inputs, hidden_layers, num_outputs):
        self.hidden_layers = hidden_layers
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.all_layers = [num_inputs] + hidden_layers + [num_outputs]
