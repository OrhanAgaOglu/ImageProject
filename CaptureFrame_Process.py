import cv2
import os
import pandas as pd
from Localization import plate_detection
from Recognize import segment_and_recognize

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
	print(vidCap.get(cv2.CAP_PROP_FPS))
	idx = 0 

	while True:
		ret, frame = vidCap.read()

		if ret == False:
			vidCap.release()
			break
		
		cv2.imwrite(f"{save_dir}/{idx}_{vidCap.get(cv2.CAP_PROP_POS_MSEC)}.png" , frame)
		idx += 1

def CaptureFrame_Process(file_path, sample_frequency, save_path):
	
	file_path = find_file_path("Video3_2.avi")
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

	for image_name in os.listdir(save_path):
		image = os.path.join(save_path, image_name)
		img = cv2.imread(image, 1)
		print(image)
		plates = plate_detection(img)
		savePlates(plates, plateFound, image_name)
		plateStrings = segment_and_recognize(plates, alphabet)
		if len(plateStrings) > 0:
			for license in plateStrings:
				if license is not None:
					print(license)
			cv2.waitKey(50)
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
		#img = cv2.resize(img, (35,50))
		char = letterPath.split("\\")[-1].split(".")[0]
		alphabet[char] = img
	for numberName in os.listdir(numberDir):
		numberPath = os.path.join(numberDir,numberName)
		img = cv2.imread(numberPath, 0)
		img = img[:,:60]
		#img = cv2.resize(img, (35,50))
		char = numberPath.split("\\")[-1].split(".")[0]
		alphabet[char] = img
	return alphabet

