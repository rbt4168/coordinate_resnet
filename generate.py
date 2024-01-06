from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import random, json
from scipy.ndimage import gaussian_filter
import tqdm
from PIL import Image
import multiprocessing as mp

def color_maping(num):
    if(num < 1/6):
        return (0, 6 * num, 1)
    elif (num < 2/6):
        return (0, 1, 1 - (num-1/6)*6)
    elif (num < 3/6):
        return ((num-2/6)*6, 1, 0)
    elif (num < 4/6):
        return (1, 1 - (num-3/6)*6, 0)
    elif (num < 5/6):
        return (1, 0, (num-4/6)*6)
    else:
        return (1 - (num-5/6)*6, 0, 1)
    

def random_color_map():
    lfmost = random.random() * 0.33
    middle = lfmost + 0.33
    rtmost = lfmost + 0.66
    outcome = [color_maping(lfmost), color_maping(middle), color_maping(rtmost)]
    random.shuffle(outcome)
    return outcome

def resize_and_save_image(input_path, output_path, new_size):
    # Open the image file
    with Image.open(input_path) as img:
        # Resize the image
        resized_img = img.resize(new_size)
        
        # Save the resized image
        resized_img.save(output_path)

def generate_gaussian_image_and_coordinates(image_number):
    # Create a 224x224 array with Gaussian noise
    values = np.zeros((224, 224)) + np.random.randn(224, 224) * 0.7 + 0.5

    # Define the boundaries for placing two random points
    lower_bound, upper_bound = 15, 224-1-15

    # Generate random coordinates for two points
    x1, y1 = random.randint(lower_bound, upper_bound), random.randint(lower_bound, upper_bound)
    x2, y2 = random.randint(lower_bound, upper_bound), random.randint(lower_bound, upper_bound)

    # Set values at random coordinates with some randomness
    values[x1, y1] = 100 * (random.random() * 1.5 + 0.5)
    values[x2, y2] = -100 * (random.random() * 1.5 + 0.5)

    # Apply Gaussian filter to smooth the image
    values = gaussian_filter(values, sigma=6)

    # Define colors for the custom colormap (R -> G -> B)
    colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]  # Red, Green, Blue

    # Create a custom colormap
    custom_colormap = LinearSegmentedColormap.from_list('custom_colormap', random_color_map(), N=256)

    # Map values to colors array using the custom colormap
    color_array = custom_colormap(values)

    # Plot contour lines of the values
    plt.clf()
    fig = plt.figure(figsize=(4,4))
    plt.axis('off')

    plt.contour(values, colors='black', linewidths=1)
    
    plt.imshow(color_array)

    # Save the generated image with a recognizable name
    plt.savefig(f"pictures/arr{image_number}.png", dpi=56, bbox_inches='tight', pad_inches=0)
    plt.close()

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
    # for i in tqdm.tqdm(range(int(2500))):
    #     image_coordinates = generate_gaussian_image_and_coordinates(i)
    #     coordinates_dict[f"arr{i}.png"] = image_coordinates
    
    # Create a pool of processes
    pool = mp.Pool(mp.cpu_count())
    n_imgs = 10000
    # Generate images and coordinates for a specified number of iterations
    for i, image_coordinates in enumerate(pool.imap(generate_gaussian_image_and_coordinates, range(int(10000)))):
        coordinates_dict[f"arr{i}.png"] = image_coordinates

    # Save the coordinates dictionary to a JSON file
    save_coordinates_to_json(coordinates_dict, "ans.json")
