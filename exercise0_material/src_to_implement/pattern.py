import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.tile_size = tile_size
        self.resolution = resolution
        self.output = None
        
    def draw(self):
        tile = self.tile_size 
        c_board = np.ones((tile * 2, tile * 2), dtype = int)
        
        c_board[:tile, :tile] = 0
        c_board[tile:, tile:] = 0
        
        a = int(self.resolution/(tile *2))
        b = np.tile(c_board,(a,a))
        self.output = b.copy()
        
        return b
    
    def show(self):
        plt.imshow(self.draw(), cmap = "gray")
        plt.show()
        
class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None
        
    def draw(self):
        x_coordinate = self.position[0]
        y_coordinate = self.position[1]
        
        c = np.linspace(0 , self.resolution - 1, self.resolution)
        a, b = np.meshgrid(c,c)
        
        c_distance = np.sqrt((a - x_coordinate)**2 + (b - y_coordinate)**2)
        
        circle = c_distance <= self.radius 
        self.output = circle.copy()
        
        return circle
    
    def show(self):
        plt.imshow(self.draw(), cmap = "gray")
        plt.show()
        
        
class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None
        
    def draw(self):
        resolution = self.resolution
        r_g_b = np.zeros([resolution,resolution, 3])
        
        r_g_b[:,:,0] = np.linspace(0,1,resolution)
        r_g_b[:,:,1] = np.linspace(0,1,resolution).reshape(resolution,1)     
        r_g_b[:,:,2] = np.linspace(1,0,resolution)

        self.output = r_g_b.copy()
        return r_g_b
        
    def show(self):
        plt.imshow(self.draw())
        plt.show()    
            