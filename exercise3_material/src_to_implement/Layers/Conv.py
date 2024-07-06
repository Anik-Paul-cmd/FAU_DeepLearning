import numpy as np
from scipy.signal import convolve2d, correlate2d
import copy
from Layers import Base

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        if isinstance(stride_shape, int):
            stride_shape = (stride_shape, stride_shape)
        elif len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        self.stride_shape = stride_shape
        self.conv2d = (len(convolution_shape) == 3)
        self.weights = np.random.uniform(size = (num_kernels, *convolution_shape))
        if self.conv2d:
            self.convolution_shape = convolution_shape
        else:
            self.convolution_shape = (*convolution_shape, 1)
            self.weights = self.weights[:, :, :, np.newaxis]    
            
        self.num_kernels = num_kernels
        self.bias = np.random.uniform(size = (num_kernels,))
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None
        self.finalShape = None
        

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, axis=-1)
            
        self.finalShape = input_tensor.shape    
 
        pad_height = self.convolution_shape[1] - 1
        pad_width = self.convolution_shape[2] - 1

        padded_image = np.zeros((input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2] + pad_height, input_tensor.shape[3] + pad_width))

        pad_height_half = self.convolution_shape[1] // 2
        pad_width_half = self.convolution_shape[2] // 2

        pad_height_even = int(self.convolution_shape[1] % 2 == 0)
        pad_width_even = int(self.convolution_shape[2] % 2 == 0)

        if pad_height_half == 0 and pad_width_half == 0:
            padded_image = input_tensor
        else:
            start_h = pad_height_half
            end_h = -pad_height_half + pad_height_even
            start_w = pad_width_half
            end_w = -pad_width_half + pad_width_even
            padded_image[:, :, start_h:end_h, start_w:end_w] = input_tensor
            
            
        input_tensor = padded_image
        self.padded = padded_image.copy()
        
        output_height = (padded_image.shape[2] - self.convolution_shape[1] +self.stride_shape[0] ) // self.stride_shape[0]
        output_width = (padded_image.shape[3] - self.convolution_shape[2]+self.stride_shape[1] ) // self.stride_shape[1]
        
        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, output_height, output_width))
        self.output_shape = output_tensor.shape
                            
        for batch_idx in range(input_tensor.shape[0]):
            for kernel_idx in range(self.num_kernels):
                for channel_idx in range(input_tensor.shape[1]):
                    correlation_result = correlate2d(input_tensor[batch_idx, channel_idx], self.weights[kernel_idx, channel_idx], mode='valid')
                    output_tensor[batch_idx, kernel_idx] += correlation_result[::self.stride_shape[0], ::self.stride_shape[1]]
                output_tensor[batch_idx, kernel_idx] += self.bias[kernel_idx]                    

        if not self.conv2d:
            output_tensor = output_tensor[:, :, :, 0] 
            
        return output_tensor 
    
    def backward(self, error_tensor):
        self.error_T = error_tensor.reshape(self.output_shape)

        if not self.conv2d:
            self.input_tensor = np.expand_dims(self.input_tensor, axis=-1)

        self.de_padded = np.zeros((self.input_tensor.shape[0], self.input_tensor.shape[1], 
                                   self.input_tensor.shape[2] + self.convolution_shape[1] - 1, 
                                   self.input_tensor.shape[3] + self.convolution_shape[2] - 1))
        self.gradient_bias = np.zeros(self.num_kernels)
        self.gradient_weights = np.zeros_like(self.weights)
        self.up_error_T = np.zeros((self.input_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2], self.input_tensor.shape[3]))
        return_tensor = np.zeros_like(self.input_tensor)

        pad_up = int(np.floor(self.convolution_shape[2] / 2))
        pad_left = int(np.floor(self.convolution_shape[1] / 2))
     
        for batch in range(self.up_error_T.shape[0]):
            h_start = pad_left
            h_end = pad_left + self.input_tensor.shape[2]
            w_start = pad_up
            w_end = pad_up + self.input_tensor.shape[3]

            self.de_padded[batch, :, h_start:h_end, w_start:w_end] = self.input_tensor[batch, :, :, :]

            for kernel in range(self.up_error_T.shape[1]):

                self.gradient_bias[kernel] += np.sum(self.error_T[batch, kernel, :])
                
                h_indices = np.arange(self.error_T.shape[2]) * self.stride_shape[0]
                w_indices = np.arange(self.error_T.shape[3]) * self.stride_shape[1]
                self.up_error_T[batch, kernel, h_indices[:, None], w_indices] = self.error_T[batch, kernel]
                
                ch = 0
                while ch < self.input_tensor.shape[1]:
                    return_tensor[batch, ch, :] += convolve2d(
                        self.up_error_T[batch, kernel, :], self.weights[kernel, ch, :], 'same')
                    ch += 1
        
            for kernel in range(self.num_kernels):
                c = 0
                while c < self.input_tensor.shape[1]:
                    self.gradient_weights[kernel, c, :] += correlate2d(
                        self.de_padded[batch, c, :], self.up_error_T[batch, kernel, :], 'valid')
                    c += 1                
                        
        if self._optimizer is not None:
            self.weights = self._optimizer.weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.bias, self.gradient_bias)

        if not self.conv2d:
            return_tensor = return_tensor[:, :, :, 0]
        return return_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weights = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)
            
    def initialize(self, weights_initializer, bias_initializer):
        total_conv_elements = np.prod(self.convolution_shape)
        self.weights = weights_initializer.initialize(self.weights.shape, total_conv_elements, np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)
