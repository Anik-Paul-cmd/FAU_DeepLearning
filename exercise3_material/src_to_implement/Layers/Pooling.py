import numpy as np
from Layers import Base

class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.finalShape = input_tensor.shape
        hight_pools = int((input_tensor.shape[2] - self.pooling_shape[0] + self.stride_shape[0]) // self.stride_shape[0])
        weight_pools = int((input_tensor.shape[3] - self.pooling_shape[1] +  self.stride_shape[1]) / self.stride_shape[1])
        output_tensor = np.zeros((input_tensor.shape[0], input_tensor.shape[1], hight_pools, weight_pools))
        self.x_axis_slice = np.zeros((input_tensor.shape[0], input_tensor.shape[1], hight_pools, weight_pools), dtype=int)
        self.y_axis_slice = np.zeros((input_tensor.shape[0], input_tensor.shape[1], hight_pools, weight_pools), dtype=int)
        
        output_row_index = -1
        i = 0         
        while i <= input_tensor.shape[2] - self.pooling_shape[0]:
            output_row_index += 1
            output_col_index = -1
            j = 0

            while j <= input_tensor.shape[3] - self.pooling_shape[1]:
                output_col_index += 1
                height_slice = slice(i, i + self.pooling_shape[0])
                width_slice = slice(j, j + self.pooling_shape[1])
                pooling_window = input_tensor[:, :, height_slice, width_slice]
                
                reshaped_window = pooling_window.reshape(*input_tensor.shape[0:2], -1)
                
                max_positions = np.argmax(reshaped_window, axis=2)
                max_pos_x, max_pos_y = divmod(max_positions, self.pooling_shape[1])
                
                self.x_axis_slice[:, :, output_row_index, output_col_index] = max_pos_x
                self.y_axis_slice[:, :, output_row_index, output_col_index] = max_pos_y
                
                batch_indices = np.arange(input_tensor.shape[0])[:, None, None]
                channel_indices = np.arange(input_tensor.shape[1])[None, :, None]
                output_tensor[:, :, output_row_index, output_col_index] = reshaped_window[
                batch_indices, channel_indices, max_positions[:, :, None]].squeeze(axis=2)
                
                j += self.stride_shape[1]
            
            i += self.stride_shape[0]        
                
        return output_tensor
    
    def backward(self, error_tensor):
        return_tensor = np.zeros(self.finalShape)
        batch_size, num_channels, slice_height, slice_width = self.x_axis_slice.shape
        stride_x, stride_y = self.stride_shape

        for batch in range(batch_size):
            for channel in range(num_channels):
                for height in range(slice_height):
                    for width in range(slice_width):
                        pos_x = self.x_axis_slice[batch, channel, height, width]
                        pos_y = self.y_axis_slice[batch, channel, height, width]
                        
                        pos_x += height * stride_x
                        pos_y += width * stride_y
                        
                        if 0 <= pos_x < self.finalShape[2] and 0 <= pos_y < self.finalShape[3]:
                            current_value = return_tensor[batch, channel, pos_x, pos_y]
                            error_value = error_tensor[batch, channel, height, width]
                            new_value = current_value + error_value
                            return_tensor[batch, channel, pos_x, pos_y] = new_value

        return return_tensor