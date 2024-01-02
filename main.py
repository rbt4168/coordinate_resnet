import math
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter
import json
import tqdm

def generate_gaussian():
    values = np.zeros((224, 224)) + np.random.randn(224, 224) * 0.7 + 0.5

    mi, Mi = 15, 224-1-15
    # Generate random coordinates
    x1, y1 = random.randint(mi, Mi), random.randint(mi, Mi)
    x2, y2 = random.randint(mi, Mi), random.randint(mi, Mi)

    # Set values at random coordinates
    values[x1, y1] = -100 * (random.random() * 1.5 + 0.5)
    values[x2, y2] = 100 * (random.random() * 1.5 + 0.5)

    # Apply Gaussian filter
    values = gaussian_filter(values, sigma=6)
    ans_arr = {"x1":x1, "y1":y1, "x2":x2, "y2":y2}
    return values, ans_arr

def generate_colormap_array():
    # 生成一個 128x128 的值在 0 到 1 之間的陣列
    values, anss = generate_gaussian()

    # Define the colors for the custom colormap (R -> G -> B)
    colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]  # Red, Green, Blue

    # Create the custom colormap
    custom_colormap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)

    # 將值映射到顏色陣列
    color_array = custom_colormap(values)

    return color_array, values, anss

def array_to_image_with_contour(array, values, num):
    # 將NumPy陣列用於等高線
    plt.contour(values, colors='black', linewidths=1)

    # 顯示圖片
    plt.imsave(f"pictures/arr{num}.png", arr=array)

def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    # 生成等高線陣列
    ax_dict = {}
    for i in tqdm.tqdm(range(int(1e4))):
        contour_array, values, anss = generate_colormap_array()
        # 將二維陣列轉換為圖片並增加等高線
        ax_dict[f"arr{i}.png"] = anss
        array_to_image_with_contour(contour_array, values, i)
        save_to_json(ax_dict, "ans.json")

