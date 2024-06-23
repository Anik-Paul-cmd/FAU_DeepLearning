import numpy as np
import matplotlib.pyplot as plt
from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

def handle_checker():
    resolution = int(input("Enter the resolution for Checker: "))
    tile_size = int(input("Enter the tile size for Checker: "))

    if resolution % (tile_size * 2) != 0:
        print("Invalid: Resolution must be divisible by 2 times the tile size.")
        return

    checker = Checker(resolution, tile_size)
    checker.show()

def handle_circle():
    resolution = int(input("Enter the resolution for Circle: "))
    radius = int(input("Enter the radius for Circle: "))
    position = tuple(map(int, input("Enter the position for Circle as 'x,y': ").split(',')))

    circle = Circle(resolution, radius, position)
    circle.show()

def handle_spectrum():
    resolution = int(input("Enter the resolution for Spectrum: "))
    spectrum = Spectrum(resolution)
    spectrum.show()

def handle_image_generator():
    file_path = "exercise_data"
    label_file = "Labels.json"
    batch_size = int(input("Enter batch size: "))
    image_size = tuple(map(int, input("Enter image size as 'height,width,channel': ").split(',')))
    rotation = input("Generate rotation images? (true/false): ").lower() == 'true'
    mirroring = input("Shuffle dataset? (true/false): ").lower() == 'true'
    shuffle = input("Apply data shuffle? (true/false): ").lower() == 'true'

    generator = ImageGenerator(file_path, label_file, batch_size, image_size, rotation, mirroring , shuffle)
    generator.show()

def main():
    part = input("Enter Exercise (press: 1) or (press: 2): ")

    if part == "1":
        sub_part = input("Enter the sub-part (1, 2, or 3): ")

        if sub_part == "1":
            handle_checker()
        elif sub_part == "2":
            handle_circle()
        elif sub_part == "3":
            handle_spectrum()
        else:
            print("Invalid sub-part. Please choose between 1, 2, or 3")
    
    elif part == "2":
        handle_image_generator()
    
    else:
        print("Invalid input. Please choose between Exercise (press: 1) or (press: 2).")

if __name__ == "__main__":
    main()
