import numpy as np
import copy

class NeuralNetwork():
    
    def __init__(self, optimizer,weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self._phase = None
        self.data_layer = None
        self.loss_layer = None
        
        self.weights_initializer = copy.deepcopy(weights_initializer)
        self.bias_initializer = copy.deepcopy(bias_initializer)
          
    @property
    def phase(self):
        return self.layers[0].testing_phase
  
    @phase.setter
    def phase(self, value):
        layer_index = 0
        while layer_index < len(self.layers):
            self.layers[layer_index].testing_phase = value
            layer_index += 1
            
    def forward(self):
        input_tensor, self.label_tensor = copy.deepcopy(self.data_layer.next())
        regularizer_loss = 0
        layer_index = 0
        while layer_index < len(self.layers):
            self.layers[layer_index].testing_phase = False
            input_tensor = self.layers[layer_index].forward(input_tensor)
            if self.optimizer.regularizer is not None and self.layers[layer_index].trainable:
                regularizer_loss += self.optimizer.regularizer.norm(self.layers[layer_index].weights)
            layer_index += 1
        
        loss = self.loss_layer.forward(input_tensor, copy.deepcopy(self.label_tensor))
        return loss + regularizer_loss
    
    def backward(self):
        error_tensor = self.loss_layer.backward(copy.deepcopy(self.label_tensor))
        
        layer_index = len(self.layers) - 1
        while layer_index >= 0:
            error_tensor = self.layers[layer_index].backward(error_tensor)
            layer_index -= 1

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)
    
    def train(self, iterations):
        self.phase = False
        iteration = 0
        while iteration < iterations:
            loss = self.forward()
            self.loss.append(loss)
            self.backward()
            iteration += 1

    def test(self, input_tensor):
        self.phase = True
        layer_index = 0
        while layer_index < len(self.layers):
            self.layers[layer_index].testing_phase = True
            input_tensor = self.layers[layer_index].forward(input_tensor)
            layer_index += 1
        return input_tensor
        