import cv2
import numpy as np

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""
def plate_detection(image):
	plate_imgs = []
    #Replace the below lines with your code.
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	colorMin = np.array([10,110,100])
	colorMax = np.array([30,250,255])
	mask = denoise(cv2.inRange(hsv, colorMin, colorMax), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
	
	#x, y, w, h = cv2.boundingRect(mask)
	output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

	# Get the number of connected components
	num_labels = output[0]
	
	# Get the label matrix
	labels = output[1]

	# Get the statistics for each connected component
	stats = output[2]

	# Iterate through each connected component
	for i in range(1, num_labels):
		# Get the bounding box coordinates for the component
		x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
		if w/h < 2:
			continue
		# Draw a rectangle around the component

		buffer = 2
		x = max(0, x - buffer)
		y = max(0, y - buffer)
		w += 2 * buffer
		h += 2 * buffer
		rotated_image = fixRotation(image[y:y+h, x:x+w])
	
		(h, w) = rotated_image.shape[:2]
		hsv = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2HSV)
		bounding_rect = cv2.boundingRect(cv2.inRange(hsv, colorMin, colorMax))
		# Crop the image using the bounding rectangle
		cropped_image = rotated_image[bounding_rect[1]:bounding_rect[1]+bounding_rect[3], bounding_rect[0]:bounding_rect[0]+bounding_rect[2]]
		plate_imgs.append(cropped_image)


	#plate_imgs = crop(mask, image)
	#plate_imgs = mask


#	if checkRatio(plate_imgs):
#		buffer = 20
#		x = max(0, x - buffer)
#		y = max(0, y - buffer)
#		w += 2 * buffer
#		h += 2 * buffer
#		plate_imgs = fixRotation(image[y:y+h, x:x+w])
#
#		return plate_imgs, True
	return plate_imgs

def checkRatio(image):
	x, y, z = image.shape
	if y< 100 or x < 33:
		return False
	return y/x >= 3.5

def denoise(img, structuring_element):
    eroded = cv2.erode(img, structuring_element)
    return cv2.dilate(eroded, structuring_element)

def fixRotation(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Perform edge detection
	edges = cv2.Canny(gray, 50, 150, apertureSize=3)
	# Run the Hough transform to detect lines in the image
	lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
	if lines is not None:	
		rho = lines[:,0,0]
		theta = lines[:,0,1]
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = x0 + 1000*(-b)
		y1 = y0 + 1000*(a)
		x2 = x0 - 1000*(-b)
		y2 = y0 - 1000*(a)
		
		lines = np.squeeze(lines)
		angles = np.arctan2(y2-y1,x2-x1)

		median_angle = np.degrees(np.median(angles))
		(h, w) = image.shape[:2]
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
		rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
		return rotated_image
	return image

