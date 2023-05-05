#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import sys
import importlib

# Add the path to the 'modules' directory to the system path
sys.path.append(r"C:\Users\rober\Downloads\CannabisClassifier\Modules")

# Load the 'modules' package
import modules

for module_file in os.listdir(modules.__path__[0]):
    if module_file.endswith('.py') and module_file != '__init__.py':
        module_name = module_file[:-3]  # Remove the '.py' extension
        module = importlib.import_module(f'{module_name}')
        
        # Import all functions and classes from the module into the global namespace
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr):
                globals()[attr_name] = attr


from preProcess import main as preprocess_main
from colorProcess import main as colorprocess_main
from analyze import main as analyze_main   



def main():
    # 1. Preprocess the images
    # This will execute all our code in Preprocess.py
    # which takes in a RAW image, converts it into PNG
    # then takes the PNG and preproccesses it and saves it 
    # eventually as a grayscale

    preprocess_main()

    # 2. Run Canny test
    # This is a test on how to make a canny edge detector
    # as the threshold settings are not immediately obvious

    canny()


    # 2.Run K-means
    # This will execute all our code in K-means.py
    # which will run K-means clustering on the flattened image
    # and then will reshape the cluster centers back into the original image shape
    # and then will save the segmented image to the output folder
    
    #kmeans() 
     
    
    # 3.
    # Color process the images
    # This will execute all our code in Colorprocess.py
    # which will show us the colors available in the thresholds
    # and then will show us the images with the colors segmented

    #colorprocess_main()

    # 4. Analyze the images
    # This will execute all our code in Analyze.py
    # which will analyze the images and show us the percentages of the colors
    # and then will show us the images with the colors segmented
    #analyze_main()
    

    #show_indexed_images(folder_path, unhealthy_indices)    #Run the process_images function from the analyze module

    #process_images()