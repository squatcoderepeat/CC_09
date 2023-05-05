import os
import glob
import rawpy
import cv2
import numpy as np
import shutil
from skimage import feature
from skimage import color
from PIL import Image

import matplotlib.pyplot as plt
import math




def convert_arw_to_png(arw_folder, png_folder):
    arw_files = glob.glob(os.path.join(arw_folder, "*.ARW"))
    os.makedirs(png_folder, exist_ok=True)

    for arw_file in arw_files:
        with rawpy.imread(arw_file) as raw:
            rgb = raw.postprocess()
        png_file = os.path.join(png_folder, os.path.splitext(os.path.basename(arw_file))[0] + '.png')
        Image.fromarray(rgb).save(png_file)

def preprocess_images(png_folder, preprocessed_folder):
    png_files = glob.glob(os.path.join(png_folder, "*.png"))
    os.makedirs(preprocessed_folder, exist_ok=True)

    for png_file in png_files:
        img = cv2.imread(png_file)

        # Resize the image
        img_resized = cv2.resize(img, (500, 500))

        # Apply Gaussian blur
        img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

        # Save the preprocessed images
        preprocessed_file = os.path.join(preprocessed_folder, os.path.basename(png_file))
        cv2.imwrite(preprocessed_file, img_blurred)

def extract_features(preprocessed_folder, save_folder):
    preprocessed_files = glob.glob(os.path.join(preprocessed_folder, "*.png"))

    for i, preprocessed_file in enumerate(preprocessed_files):
        img = cv2.imread(preprocessed_file)

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save the grayscale image
        gray_file = os.path.join(save_folder, os.path.basename(preprocessed_file))
        cv2.imwrite(gray_file, gray_img)

        # Extract the histogram of oriented gradients (HOG) feature
        hog_feature = feature.hog(gray_img)

        # Gaussian blur
        blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        # Save the blurred image
        blur_file = os.path.join(save_folder, os.path.splitext(os.path.basename(preprocessed_file))[0] + '_blur.png')
        cv2.imwrite(blur_file, blur)

        edges = cv2.Canny(gray_img, 100, 200)

        # Save the edges image
        edges_file = os.path.join(save_folder, os.path.splitext(os.path.basename(preprocessed_file))[0] + '_edges.png')
        cv2.imwrite(edges_file, edges)

        # Compute the color histogram
        color_hist = color.rgb2gray(img).flatten()

        # Convert the image to HSV color space to identify the location of different colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Save the HSV image
        hsv_file = os.path.join(save_folder, os.path.splitext(os.path.basename(preprocessed_file))[0] + '_hsv.png')
        cv2.imwrite(hsv_file, hsv)

        # Combine the features
        combined_features = np.concatenate((hog_feature, color_hist))
        num_features = len(hog_feature) + len(color_hist)
        
        if i == 0:
            global features_array
            features_array = np.empty((len(preprocessed_files), num_features))
                
        features_array[i] = combined_features
            
    return features_array




def show_generated_images(folder_path):
    image_files = glob.glob(os.path.join(folder_path, "*.png"))
    
    num_images = len(image_files)
    
    if num_images == 0:
        print("No images found in the folder.")
        return

    num_rows = int(math.sqrt(num_images))
    num_cols = math.ceil(num_images / num_rows)
    
    plt.figure(figsize=(20, 20))
    
    for i, image_file in enumerate(image_files, start=1):
        img = Image.open(image_file)
        
        title = os.path.splitext(os.path.basename(image_file))[0].split("_")[-1]
        title = title.split('.')[0] # get the part between _ and .png
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(folder_path, title +'.png'))
    plt.show()


def main():

    # I changed the way the folders are chosen now the 
    # user can choose the folder where the arw files are
    # and the program will create the png folder inside


    arw_folder = input("Enter input folder path: ")
    png_folder = os.path.join(arw_folder, 'png')
    os.makedirs(png_folder, exist_ok=True)

    convert_arw_to_png(arw_folder, png_folder)

    preprocessed_folder = os.path.join(png_folder, 'preprocess')
    # I liked to have a lot of folders so that I can see each step of the process
    preprocess_images(png_folder, preprocessed_folder)

    save_folder = os.path.join(png_folder, 'preprocess_images')
    os.makedirs(save_folder, exist_ok=True)

    features_array = extract_features(preprocessed_folder, save_folder)
    show_generated_images(save_folder)

    # Save the features array to an .npy file
    np.save(os.path.join(save_folder, 'array_preprocess.npy'), features_array)

    # Load the array from the .npy file
    array = np.load(os.path.join(save_folder, 'array_preprocess.npy'))
    print(f"Features array saved to: {os.path.abspath(os.path.join(save_folder, 'array_preprocess.npy'))}")

    # Move the grayscale blurred images to a separate folder
    source_folder = os.path.join(preprocessed_folder, 'preprocessed_step2')
    destination_folder = os.path.join(source_folder, 'gray')

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Loop through all the files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file ends with "_blur.png"
        if filename.endswith("_blur.png"):
            # Construct the file paths for the source and destination
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)

            # Move the file from the source folder to the destination folder
            shutil.move(source_path, destination_path)


if __name__ == "__main__":
    main()
