import cv2
import numpy as np
import os

"""
In this file, you will define your own segment_and_recognize function.
To do:
	1. Segment the plates character by character
	2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
	3. Recognize the character by comparing the distances
Inputs:(One)
	1. plate_imgs: cropped plate images by Localization.plate_detection function
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
	1. recognized_plates: recognized plate characters
	type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
	You may need to define other functions.
"""
def segment_and_recognize(plate_imgs, alphabet):
	if len(plate_imgs)==0:
		return np.empty(0)
	
	recognized_plates = np.empty(len(plate_imgs), dtype=object)
	for index, plate in enumerate(plate_imgs):
		string = ""
		charaters = segmentation(plate)
		for char in charaters:
			#cv2.imshow("mask2", char)
			#cv2.waitKey(2000)
			c = recognition(char,alphabet)
			string = string + c 
		if(len(string)==6):
			recognized_plates[index] = string
		else:
			recognized_plates[index] = ""
	return recognized_plates

def segmentation(plate):
	colorMin = np.array([10,125,115])
	colorMax = np.array([30,250,255])
	height, width = plate.shape[:2]
	new_width = int(width * 0.96)
	hsv = cv2.cvtColor(plate[:height, :new_width], cv2.COLOR_BGR2HSV)
	median = cv2.medianBlur(hsv, 3)
	mask = cv2.inRange(median, colorMin, colorMax)
	#cv2.imshow("mask2", mask)
	#cv2.waitKey(2000)
	wbRatio = whiteBlackRatio(mask)
	if wbRatio <2.2 or wbRatio >3.5:
		gray = hsv[:,:,2]
		equalized = cv2.equalizeHist(gray)
		#cv2.imshow("mask", equalized)
		#cv2.waitKey(1000)
		_, binary = cv2.threshold(equalized, 70, 255, cv2.THRESH_BINARY)
		
		#cv2.imshow("BINARIZED MEDIAN", binary)
		#cv2.waitKey(1000)
		mask = binary
	#binary = denoise(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
	
	#threshold, thresholded_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	flood_fill_image = mask.copy()
	h, w = flood_fill_image.shape[:2]
	mask2 = np.zeros((h+2, w+2), np.uint8)

	# Flood fill the background
	cv2.floodFill(flood_fill_image, mask2, (0,0), 255)
	
	# Invert the flood fill image
	flood_fill_image = cv2.bitwise_not(flood_fill_image)
	#cv2.imshow("FLOOOD FILLLLLLL!L!!P{!KL!KO!K!O!", flood_fill_image)
	#cv2.waitKey(200)
	contours, _ = cv2.findContours(flood_fill_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
	characters = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		character = flood_fill_image[y:y+h, x:x+w]
		if checkCharacterRatio(character, contour):
			characters.append(character)
	return characters

def recognition(char, alphabet):
	# Compare the character with each alphabet image
	letter = alphabet["B"]
	#charResized = cv2.resize(char, (letter.shape[1] , letter.shape[0]))
	charResized = resize_and_concatenate(char, letter)
	closest_match = None
	closest_match_diff = float("inf")
	for alphabet_index, alphabet_image in alphabet.items():
		diff = cv2.norm(charResized, alphabet_image, cv2.NORM_L1)
		if diff < closest_match_diff:
			closest_match_diff = diff
			closest_match = alphabet_index
	#print("Character is most likely the letter {}".format(closest_match))
	return closest_match
	
	

def find_file_path(file_name):
    for root, dirs, files in os.walk("."):
        if file_name in files:
            return os.path.join(root, file_name)
    return None
def checkCharacterRatio(character, contour):
	x,y,w,h = cv2.boundingRect(contour)
	if h/w >1.2 and w>3:
		return True
	return False
def denoise(img, structuring_element):
    eroded = cv2.erode(img, structuring_element)
    return cv2.dilate(eroded, structuring_element)
def resize_and_concatenate(image, reference_image):
	# Get the original image size
	height, width = image.shape[:2]

	# Get the reference image size
	ref_height, ref_width = reference_image.shape[:2]

	# Calculate the new width based on the height of the reference image and the aspect ratio of the original image
	new_width = int(width * ref_height / height)

	# Resize the image
	resized_image = cv2.resize(image, (new_width, ref_height))

	# Create a black image with the same size as the reference image
	black_image = np.zeros((ref_height, np.abs(ref_width - new_width)), np.uint8)

	# Concatenate the resized image and the black image to get the final image
	final_image = np.concatenate((resized_image, black_image), axis=1)
	final_image = cv2.resize(final_image, (ref_width, ref_height))
	return final_image
def whiteBlackRatio(img):
	white_pixels = cv2.countNonZero(img)
	img_invert = cv2.bitwise_not(img)
	black_pixels = cv2.countNonZero(img_invert)
	if black_pixels == 0:
		return 0
	return white_pixels / black_pixels
	