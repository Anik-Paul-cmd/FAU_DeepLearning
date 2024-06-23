import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        
        with open(self.label_path, 'r') as f:
            self.class_labels = json.load(f)
            
        self.__sample_size = len(self.class_labels)
        self.__epoch = 0
        self.__batch_number = 0
        if self.__sample_size < self.batch_size or self.batch_size == 0:
            self.batch_size = self.__sample_size
        self.__sample_indices = np.arange(self.__sample_size)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        if self.__batch_number * self.batch_size >= self.__sample_size:
            self.__epoch += 1
            self.__batch_number = 0
            
        if self.__batch_number == 0 and self.shuffle:
            np.random.shuffle(self.__sample_indices)

        images = np.zeros((self.batch_size, *tuple(self.image_size)))
        labels = np.zeros(self.batch_size, dtype = int)
        start_index = self.__batch_number * self.batch_size

        index = 0
        while(index < self.batch_size and start_index+index < self.__sample_size):
            images[index] = self.augment(np.load(f"{self.file_path}/{self.__sample_indices[start_index+index]}.npy"))
            labels[index] = self.class_labels[str(self.__sample_indices[start_index+index])]
            index+=1
            print(index)
        current_batch_size = index
        
        index = 0
        while(current_batch_size + index < self.batch_size):
            images[current_batch_size + index] = self.augment(np.load(f"{self.file_path}/{self.__sample_indices[index]}.npy"))
            labels[current_batch_size + index] = self.class_labels[str(self.__sample_indices[index])]
            
            index+=1
            print('-----')
            print(index)
            
        self.__batch_number+=1    
        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if img.shape != self.image_size:
            img = np.resize(img, self.image_size)
        if self.mirroring:
            if np.random.choice((True, False)):
                img = np.fliplr(img)
            if np.random.choice((True, False)):
                img = np.flipud(img)    
        if self.rotation:
            n = np.random.randint(4)
            img = np.rot90(img, n)        
        return img
        

    def current_epoch(self):
        # return the current epoch number
        return self.__epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_labels[str(x)]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        image, label = self.next()
        fig = plt.figure(figsize=(10,10))
        if self.batch_size % 3:
            remainder = 1
        else:
            remainder = 0
        column = 3
        row = self.batch_size // 3 + remainder
        for i in range(1, self.batch_size+1):
            fig.add_subplot(row, column, i)
            plt.imshow(image[i-1].astype('uint8'))
            plt.xticks([])
            plt.yticks([])
            plt.title(self.class_dict[label[i-1]])
        plt.show()

x = ImageGenerator("exercise_data", "Labels.json", 30, (32,32,3), True, True, True)

for i in range(4):
    x.next()
    print('-----')