from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import random, json
from scipy.ndimage import gaussian_filter
import tqdm

def generate_gaussian_image_and_coordinates(image_number):
    # Create a 224x224 array with Gaussian noise
    values = np.zeros((224, 224)) + np.random.randn(224, 224) * 0.7 + 0.5

    # Define the boundaries for placing two random points
    lower_bound, upper_bound = 15, 224-1-15

    # Generate random coordinates for two points
    x1, y1 = random.randint(lower_bound, upper_bound), random.randint(lower_bound, upper_bound)
    x2, y2 = random.randint(lower_bound, upper_bound), random.randint(lower_bound, upper_bound)

    # Set values at random coordinates with some randomness
    values[x1, y1] = -100 * (random.random() * 1.5 + 0.5)
    values[x2, y2] = 100 * (random.random() * 1.5 + 0.5)

    # Apply Gaussian filter to smooth the image
    values = gaussian_filter(values, sigma=6)

    # Define colors for the custom colormap (R -> G -> B)
    colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]  # Red, Green, Blue

    # Create a custom colormap
    custom_colormap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)

    # Map values to colors array using the custom colormap
    color_array = custom_colormap(values)

    # Plot contour lines of the values
    plt.contour(values, colors='black', linewidths=1)

    # Save the generated image with a recognizable name
    plt.imsave(f"pictures/arr{image_number}.png", arr=color_array)

    # Store coordinates in a dictionary for each image
    coordinates = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    return coordinates

def save_coordinates_to_json(data, filename):
    # Save coordinates dictionary to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    # Dictionary to store image coordinates
    coordinates_dict = {}

    # Generate images and coordinates for a specified number of iterations
    for i in tqdm.tqdm(range(int(100))):
        image_coordinates = generate_gaussian_image_and_coordinates(i)
        coordinates_dict[f"arr{i}.png"] = image_coordinates

    # Save the coordinates dictionary to a JSON file
    save_coordinates_to_json(coordinates_dict, "ans.json")
