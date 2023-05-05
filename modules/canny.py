

#This is a test on how to make a canny edge detector
# as the threshold settings are not immediately obvious
# therefore this can be run optionally, to figure out what threshold is best,
# and then the best threshold can be used in the main program
# however a lot of images are created.

def canny_test(preprocessed_folder, output_folder):
    high_threshold = 255
    low_threshold = high_threshold / 3

    highThreshold_range = range(255//3, 256, 10)
    array_ab = [(low, high) for high in highThreshold_range for low in range(high//3, high, 10)]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(preprocessed_folder):
        image_path = os.path.join(preprocessed_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        for a, b in array_ab:
            output_path = os.path.join(output_folder, f'{filename}-canny{a}_{b}.png')

            # Check if the output image already exists
            if os.path.exists(output_path):
                continue

            edges = cv2.Canny(image, a, b)
            cv2.imwrite(output_path, edges)

    show_generated_images(output_folder)

