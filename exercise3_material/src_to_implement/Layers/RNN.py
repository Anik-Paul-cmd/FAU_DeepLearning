import numpy as np
import copy 
from Layers import Base
from Layers import FullyConnected as FC
from Layers import Sigmoid
from Layers import TanH

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self._memorize = False
        self._optimizer = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc_hidden = FC.FullyConnected(input_size+hidden_size, hidden_size)
        self.hidden_state = [np.zeros(self.hidden_size)]
        self.tanh = TanH.TanH()
        self.sigmoid = Sigmoid.Sigmoid()
        self.fc_output = FC.FullyConnected(hidden_size, output_size)
        self.gradient_hidden_weights = None
        self.gradient_output_weights = None
        self._gradient_weights = self.gradient_hidden_weights
        self._weights = self.fc_hidden.weights

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output = np.zeros((self.input_tensor.shape[0], self.output_size))
        if not self._memorize:
            self.hidden_state = [np.zeros((1, self.hidden_size))]
        t = 0
        for i in input_tensor:
            x_t = i[np.newaxis, :]
            h_t_1 = self.hidden_state[-1].flatten()[np.newaxis, :]
            concatenated_input = np.concatenate((x_t, h_t_1), axis=1)
            self.hidden_state.append(self.tanh.forward(self.fc_hidden.forward(concatenated_input)))
            self.output[t] = self.sigmoid.forward(self.fc_output.forward(self.hidden_state[-1]))
            t += 1
            
        return self.output

    def backward(self, error_tensor):
        self.gradient_hidden_weights = np.zeros(self.fc_hidden.weights.shape)
        self.gradient_output_weights = np.zeros(self.fc_output.weights.shape)
        output_error = np.zeros((self.input_tensor.shape[0], self.input_size))
        error_h_t = np.zeros((1, self.hidden_size))

        i = error_tensor.shape[0]
        while i > 0:
            i -= 1
            x_t = self.input_tensor[i][np.newaxis, :]  
            h_t_1 = self.hidden_state[i].flatten()[np.newaxis, :]  
            concatenated_input = np.concatenate((x_t, h_t_1), axis=1)
            hidden_layer_out = self.fc_hidden.forward(concatenated_input)
            tanh_output = self.tanh.forward(hidden_layer_out)
            fc_out_out = self.fc_output.forward(tanh_output)
            self.sigmoid.forward(fc_out_out)
            gradient = self.fc_hidden.backward(self.tanh.backward(self.fc_output.backward(self.sigmoid.backward(error_tensor[i, :])) + error_h_t))
            self.gradient_hidden_weights += self.fc_hidden.gradient_weights
            self.gradient_output_weights += self.fc_output.gradient_weights
            output_error[i], error_h_t = gradient[:, :self.input_size].copy(), gradient[:, self.input_size:].copy()

        if self.optimizer:
            self.fc_hidden.weights = self.optimizer.calculate_update(self.fc_hidden.weights, self.gradient_hidden_weights)
            self.fc_output.weights = self.optimizer.calculate_update(self.fc_output.weights, self.gradient_output_weights)

        return output_error

    @property
    def memorize(self):
        return self.memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize= memorize

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, weights):
        self.fc_hidden.weights = weights
        
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)    

    @property
    def gradient_weights(self):
        return self.gradient_hidden_weights

    @gradient_weights.setter
    def gradient_weights(self, new_weights):
        self.fc_hidden._gradient_weights = new_weights


        
        
        




