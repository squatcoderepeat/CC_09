import os
import cv2
import numpy as np
from pathlib import Path
from skimage import io
from sklearn.cluster import KMeans

###################################################
def analyze_plant_health(image_path, num_clusters=8):
    segmented_img = cv2.imread(image_path)
    color_percentages = {}

    if segmented_img is not None:
        unique_colors, percentages = get_color_percentages(segmented_img, num_clusters)  # Change function name

        for color, percentage in zip(unique_colors, percentages):
            for lower_range, upper_range, color_name in color_ranges:
                if (color >= lower_range).all() and (color <= upper_range).all():
                    color_category = color_name
                    break
            else:
                color_category = 'Other'

            color_percentages[color_category] = percentage

    else:
        print(f"Image could not be read for file '{image_path}'")

    return color_percentages


# Define a function to check the health of the plant
def is_plant_healthy(color_percentages, thresholds):
    problems = []

    if color_percentages[1] > thresholds['yellow']:
        problems.append('Yellowing leaves (possible nutrient deficiency)')

    if color_percentages[2] > thresholds['brown']:
        problems.append('Browning leaves (possible nutrient burn or over-watering)')

    if color_percentages[3] > thresholds['purple']:
        problems.append('Purpling leaves (possible phosphorus deficiency or cold stress)')

    return problems

def main():
    process_images()


    # Example usage
    input_folder = input("Enter input folder path: ")
    kmeans_folder = os.path.join(kmeans_folder, 'png')
    os.makedirs(kmeans_folder, exist_ok=True)

    image_color_percentages = analyze_plant_health(input_folder, num_clusters=4)

    color_names = [color_info[2] for color_info in color_ranges]
    color_names.append('Other')

    array_data = np.zeros((len(image_color_percentages), len(color_names)))

    for i, color_percentages in enumerate(image_color_percentages):
        for j, color_name in enumerate(color_names):
            array_data[i, j] = color_percentages.get(color_name, 0)

    print("Color percentages array:")
    print(array_data)


    # Define threshold values for color ranges (in percentages)
    thresholds = {
        'yellow': 10,
        'brown': 5,
        'purple': 5
    }


    # Initialize arrays for healthy and unhealthy indices
    healthy_indices = []
    unhealthy_indices = []

    # Iterate through the images and check if the plant is healthy or not
    for idx, color_percentages in enumerate(array_data):
        problems = is_plant_healthy(color_percentages, thresholds)
        if problems:
            unhealthy_indices.append(idx)
            print(f"Image index {idx} has the following problems:")
            for problem in problems:
                print(f"  - {problem}")
        else:
            healthy_indices.append(idx)
            print(f"Image index {idx} appears to be healthy.")

    # Print the indices of healthy and unhealthy plants
    print("Healthy indices:", healthy_indices)
    show_indexed_images(folder_path, healthy_indices)

    # Show the unhealthy images
    print("Unhealthy indices:", unhealthy_indices)
    show_indexed_images(folder_path, unhealthy_indices)




if __name__ == "__main__":
    main()