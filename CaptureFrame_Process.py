import cv2
import os
import pandas as pd
from Localization import plate_detection
from Recognize import segment_and_recognize
import numpy as np
from scipy.spatial import distance


"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""
def create_dir(path): 
	try:
		if not os.path.exists(path):
			os.makedirs(path)
	except OSError:
		print(f"ERROR: directory with name{path} already exists")

def capture_frame(video_path, fps, save_dir):

	vidCap = cv2.VideoCapture(video_path)
	vidCap.set(cv2.CAP_PROP_FPS, fps)
	idx = 0 

	while True:
		ret, frame = vidCap.read()
		if ret == False:
			vidCap.release()
			break
		timestamp = vidCap.get(cv2.CAP_PROP_POS_MSEC)
		time = timestamp / 1000
		cv2.imwrite(f"{save_dir}/{time:.2f}.png" , frame)
		idx += 1

def CaptureFrame_Process(file_path, sample_frequency, save_path):
	
	file_path = find_file_path("Video31_2.avi")
	save_path = os.path.dirname(file_path)
	sample_frequency = 1
	
	name = file_path.split("\\")[-1].split(".")[0]
	save_path = os.path.join(save_path,name)	
	create_dir(save_path)
	plateFound = save_path+"\detected_plates"
	create_dir(plateFound)
	capture_frame(file_path, sample_frequency, save_path)
	lettersPath = os.path.dirname(find_file_path("B.bmp"))
	numbersPath = os.path.dirname(find_file_path("1.bmp"))
	alphabet = createAlphabet(lettersPath, numbersPath)

	guesses = []
	for frame_no,image_name in enumerate(os.listdir(save_path)):
		if(image_name == "detected_plates"):
			continue
		image = os.path.join(save_path, image_name)
		img = cv2.imread(image, 1)
		plates = plate_detection(img)
		plates_groups = np.empty((2,len(plates)))
		group_index = 1
		for plate_index, plate in enumerate(plates):
			if plates_groups.size > 0:
				threshold = 100
				distance = hammingDistance(plate,plates[plate_index - 1])
				if distance > threshold:
					group_index += 1
			plates_groups[plate_index] = [group_index, plate]
		#print(plates_groups)
		savePlates(plates, plateFound, image_name)
		plateStrings = segment_and_recognize(plates, alphabet)
		for plate in plateStrings:
			if plate != "":
				timestamp = image_name.replace(".png", "")
				guesses.append([plate, frame_no, timestamp])
	mostLikely = getBestString(guesses)
	array = []
	result = get_closest_match(mostLikely, guesses)
	array.append(result)
	save_as_csv(array, "output.csv")

def find_file_path(file_name):
    for root, dirs, files in os.walk("."):
        if file_name in files:
            return os.path.join(root, file_name)
    return None

def savePlates(plates, plateFound, image_name):
	for plate in plates:
		cv2.imwrite(plateFound + "\\" + f"{image_name}" + ".png" , plate)

def createAlphabet(letterDir, numberDir):
	alphabet = {}
	for letterName in os.listdir(letterDir):
		letterPath = os.path.join(letterDir,letterName)
		img = cv2.imread(letterPath, 0)
		img = img[:,:60]
		#positiveRotImg = RotateImage(img, 5)
		#positiveRotImg = cv2.resize(positiveRotImg, img.shape)
		#negativeRotImg = RotateImage(img, 355)
		#negativeRotImg = cv2.resize(negativeRotImg, img.shape)
		#cv2.imshow("positiveRotImg", positiveRotImg)
		#cv2.waitKey(2000)
		#cv2.imshow("negativeRotImg",negativeRotImg)
		#cv2.waitKey(2000)
		#img = cv2.resize(img, (35,50))
		char = letterPath.split("\\")[-1].split(".")[0]
		alphabet[char] = img
		#alphabet[char+"_PositiveRot"] = positiveRotImg
		#alphabet[char+"_NegativeRot"] = negativeRotImg
	for numberName in os.listdir(numberDir):
		numberPath = os.path.join(numberDir,numberName)
		img = cv2.imread(numberPath, 0)
		img = img[:,:60]
		#positiveRotImg = RotateImage(img, 5)
		#positiveRotImg = cv2.resize(positiveRotImg, img.shape)
		#negativeRotImg = RotateImage(img, 355)
		#negativeRotImg = cv2.resize(negativeRotImg, img.shape)
		#cv2.imshow("positiveRotImg", positiveRotImg)
		#cv2.imshow("negativeRotImg",negativeRotImg)
		#img = cv2.resize(img, (35,50))
		char = numberPath.split("\\")[-1].split(".")[0]
		alphabet[char] = img
		#alphabet[char+"_PositiveRot"] = positiveRotImg
		#alphabet[char+"_NegativeRot"] = negativeRotImg
	return alphabet

def RotateImage(Image, angle):
    
	imgHeight, imgWidth = Image.shape[0], Image.shape[1]

	centreY, centreX = imgHeight//2, imgWidth//2

	rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), angle, 1.0)
	cosofRotationMatrix = np.abs(rotationMatrix[0][0])
	sinofRotationMatrix = np.abs(rotationMatrix[0][1])

	newImageHeight = int((imgHeight * sinofRotationMatrix) +
						(imgWidth * cosofRotationMatrix))
	newImageWidth = int((imgHeight * cosofRotationMatrix) +
						(imgWidth * sinofRotationMatrix))

	rotationMatrix[0][2] += (newImageWidth/2) - centreX
	rotationMatrix[1][2] += (newImageHeight/2) - centreY

	rotatingimage = cv2.warpAffine(
		Image, rotationMatrix, (newImageWidth, newImageHeight))

	return rotatingimage

# Method to return the most likely string based on
# the frequency of the characters in each index
def getBestString(strings):
    strings = [x[0] for x in strings if x != '']
    # rest of the code remains the same
    num_chars = len(strings[0])
    most_frequent_chars = np.empty(num_chars, dtype=str)

    for i in range(num_chars):
        chars_at_index = [string[i] for string in strings]
        unique_chars, counts = np.unique(chars_at_index, return_counts=True)
        max_count_index = np.argmax(counts)
        most_frequent_chars[i] = unique_chars[max_count_index]

    final_string = "".join(most_frequent_chars)
    return final_string


# Saver method to construct the output
def save_as_csv(data, filename):
	with open(filename, "w") as f:
		f.write("License plate"+ ","+ "Frame no."+","+"Timestamp(seconds)" + "\n")
		for line in data:
			result = f"{line[0]},{line[1]},{float(line[2]):.4f}"
			f.write(result +"\n")

# Method to transform string of plate into approporiate format
def format_license_plate(plate):
	formatted_plate = ""
	countLetter = 0
	countNumber = 0
	for i in range(len(plate)):
		if (i > 0 and (plate[i].isalpha() != plate[i-1].isalpha())):
			formatted_plate += "-"
		formatted_plate += plate[i]
	for j in range(len(formatted_plate)):
		if formatted_plate[j] == "-":
			continue
		if formatted_plate[j].isalpha():
			countLetter += 1
			countNumber = 0
		else: 
			countNumber += 1
			countLetter = 0
		if(countLetter ==4 or countNumber ==4):
			formatted_plate = formatted_plate [:j-1] + '-' + formatted_plate [j-1:]
			countLetter = 0
			countNumber = 0
	return formatted_plate

def get_closest_match(input_string, strings):
	distances = [levenshtein(input_string, string) for string in strings]
	min_distance = min(distances)
	min_distance_index = distances.index(min_distance)
	closest_string = strings[min_distance_index][0]
	result=[]
	for string in strings:
		if string[0] == closest_string:
			result= string
	result[0] = format_license_plate(result[0])
	return result

def levenshtein(s1, s2):
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    if s1[-1] == s2[-1]:
        cost = 0
    else:
        cost = 1
    return min(levenshtein(s1[:-1], s2) + 1,
               levenshtein(s1, s2[:-1]) + 1,
               levenshtein(s1[:-1], s2[:-1]) + cost)

def hammingDistance(img1, img2):
	bgr1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	bgr2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	threshold1 = cv2.threshold(bgr1, 128, 255, cv2.THRESH_BINARY)
	threshold2 = cv2.threshold(bgr2, 128, 255, cv2.THRESH_BINARY)
	threshold1 = np.array(threshold1)
	threshold2 = np.array(threshold2)
	flat_1 = threshold1.flatten()
	flat_2 = threshold2.flatten()
	#diff = cv2.bitwise_xor(threshold1, threshold2)
	#hamming_distance = cv2.countNonZero(diff)
	ham_dist = distance.hamming(flat_1, flat_2)
	return ham_dist