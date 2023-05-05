from skimage import io
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Define the color ranges for each color
green_lower = np.array([20, 20, 10])
green_upper = np.array([80, 220, 80])
yellow_lower = np.array([190, 190, 80])
yellow_upper = np.array([255, 255, 130])
brown_lower = np.array([70, 40, 30])
brown_upper = np.array([180, 140, 105])
purple_lower = np.array([60, 40, 55])
purple_upper = np.array([200, 110, 200])
color_ranges = [(green_lower, green_upper), (yellow_lower, yellow_upper), (brown_lower, brown_upper), (purple_lower, purple_upper)]
colors = ['Green', 'Yellow', 'Brown', 'Purple']

    # Normalize the color ranges to the range of 0-1
green_lower_rgb = green_lower / 255.0
green_upper_rgb = green_upper / 255.0
yellow_lower_rgb = yellow_lower / 255.0
yellow_upper_rgb = yellow_upper / 255.0
brown_lower_rgb = brown_lower / 255.0
brown_upper_rgb = brown_upper / 255.0
purple_lower_rgb = purple_lower / 255.0
purple_upper_rgb = purple_upper / 255.0



#We need to actually see the colors we are thresholding
#Therefore a colormap will help us visualize the colors


def colormap():
    # Create color map
    colors = [green_lower_rgb, green_upper_rgb, yellow_lower_rgb, yellow_upper_rgb, brown_lower_rgb, brown_upper_rgb, purple_lower_rgb, purple_upper_rgb]
    cmap = ListedColormap(colors)

    # Create figure with color bar
    fig, ax = plt.subplots(figsize=(10, 1))
    cb = plt.colorbar(ax.imshow(np.arange(len(colors)).reshape(1, len(colors)), cmap=cmap, aspect='auto'),
                      cax=ax, ticks=range(len(colors)))
    cb.ax.set_yticklabels(['green_lower', 'green_upper', 'yellow_lower', 'yellow_upper', 'brown_lower', 'brown_upper', 'purple_lower', 'purple_upper'])
    plt.show()

# Define function process_images() to process a set of images by finding contours, 
# segmenting the images based on color ranges,
# and saving the segmented images and contours to an output folder.

def process_images():
    output_folder = '/output_images/'
    
    output_path = os.path.join(output_folder, preprocessed_folder)
    
    
    os.makedirs(output_folder, exist_ok=True)

    for i in range(10, 50):
        print(f"Processing image {i}")
        file_path = preprocessed_folder
        print(f"File path: {file_path}")
        img = io.imread(file_path)

        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]


            for j, color_range in enumerate(color_ranges):
                lower_range = np.array(color_range[0])
                upper_range = np.array(color_range[1])
                mask = cv2.inRange(hsv, lower_range, upper_range)
                segmented = cv2.bitwise_and(img, img, mask=mask)
                cv2.imwrite(f'{output_folder}cannabis-{i}-{colors[j]}-Segmented.png', segmented)

            output_path = f'{output_folder}cannabis-{i}-Contour.png'
            cv2.imwrite(output_path, img)
            print(f"Image saved at: {output_path}")
            print("Image processing complete")
        else:
            print(f"Image could not be read for file '{file_path}'")


#get_color_percentages(image, num_clusters) takes an image and the number of desired clusters as inputs, 
#and applies K-means clustering to the pixels of the image to obtain the dominant colors and their percentages. 
#he function returns the dominant colors in RGB values and their corresponding percentages.


def get_color_percentages(image, num_clusters):
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Apply K-means clustering to the pixels
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)

    # Count the number of pixels in each cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    # Calculate the percentage of pixels in each cluster
    percentages = [count / sum(pixel_counts.values()) * 100 for count in pixel_counts.values()]

    return kmeans.cluster_centers_.astype(int), percentages



def main():
    process_images()
    # Call the colormap function
    colormap()


