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
	for plate in plate_imgs:
		charaters = segmentation(plate)
		for char in charaters:
			cv2.imshow("char", char)
			cv2.waitKey(200)
			recognition(char,alphabet)
	return recognized_plates

def segmentation(plate):
	colorMin = np.array([10,110,100])
	colorMax = np.array([30,250,255])
	hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, colorMin, colorMax)
	kernel = np.ones((2,2), np.uint8)
	#dilated_mask = cv2.dilate(mask, kernel, iterations=1)
	#threshold, thresholded_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	flood_fill_image = mask.copy()
	h, w = flood_fill_image.shape[:2]
	mask2 = np.zeros((h+2, w+2), np.uint8)

	# Flood fill the background
	cv2.floodFill(flood_fill_image, mask2, (0,0), 255)

	# Invert the flood fill image
	flood_fill_image = cv2.bitwise_not(flood_fill_image)
	contours, _ = cv2.findContours(flood_fill_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
	characters = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		character = plate[y:y+h, x:x+w]
		if checkCharacterRatio(character, contour):
			characters.append(character)
	return characters

def recognition(char, alphabet):
	# Compare the character with each alphabet image
	letter = alphabet["B"]
	charResized = cv2.resize(char, (35,50))
	colorMin = np.array([10,110,100])
	colorMax = np.array([30,250,255])
	hsv = cv2.cvtColor(char, cv2.COLOR_BGR2HSV)
	charResized = cv2.bitwise_not(cv2.inRange(hsv, colorMin, colorMax))
	charResized = denoise(charResized, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))
	charResized = cv2.resize(charResized, (letter.shape[1] , letter.shape[0]))
	closest_match = None
	closest_match_diff = float("inf")
	cv2.imshow("r", charResized)
	cv2.waitKey(200)
	for alphabet_index, alphabet_image in alphabet.items():
		diff = cv2.norm(charResized, alphabet_image, cv2.NORM_L1)
		if diff < closest_match_diff:
			closest_match_diff = diff
			closest_match = alphabet_index
	print("Character is most likely the letter {}".format(closest_match))
	
	

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