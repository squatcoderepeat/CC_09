import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

def kmeans(num_clusters=8,input_folder):
    # Define the source and destination paths
    folder_path_1 = Path(r'C:\Users\rober\Pictures\mar 21\purple !!!\preprocessed_step2')
    output_folder = folder_path_1 / "kmeans_output"

    # Create the output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Loop over each file in the source folder
    for filename in folder_path_1.iterdir():
        if filename.name.endswith('hsv.png'):
            # Load the preprocessed image
            image = cv2.imread(str(filename))

            # Flatten the image into a 2D array of pixels
            pixels = np.reshape(image, (-1, 3))

            # Run KMeans clustering on the flattened image
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(pixels)

            # Reshape the cluster centers back into the original image shape
            cluster_centers = kmeans.cluster_centers_.astype('uint8')
            segmented_image = cluster_centers[kmeans.labels_]
            segmented_image = np.reshape(segmented_image, image.shape)

            # Save the segmented image to the output folder
            output_path = output_folder / f"{filename.stem}-KMeans-Segmented.png"
            cv2.imwrite(str(output_path), segmented_image)

            edges = cv2.Canny(segmented_image, 100, 200)
            output_path = output_folder / f"{filename.stem}-KMeans-Segmented-edges.png"
            cv2.imwrite(str(output_path), edges)


if __name__ == "__main__":
    kmeans()
