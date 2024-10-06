import cv2
import numpy as np
from scipy import ndimage
import skimage

from feed import VideoFeed


# Compute frame difference between the previous frame and the current ont
def compute_frame_difference(gray: np.ndarray):
	global prev_frame

	if prev_frame is None:
		prev_frame = gray
		return np.zeros_like(gray)

	signed_gray = np.astype(gray, np.int16)
	result = np.clip(signed_gray - prev_frame, 0, 255)

	prev_frame = gray
	return np.astype(result, np.uint8)


# Apply thresholding to the movement to filter out small changes
def apply_thresholding(difference: np.ndarray):
	thresh = 45
	return np.array(np.where(difference > thresh, 255, 0), dtype=np.uint8)


# Label different contours for multiple intrusion points detection
def label_image(difference):
	s = np.ones((3, 3))  # 8-connectivity
	labelled, groups = ndimage.label(difference, s)
	return np.array(labelled, dtype=np.uint8)


# Erode the labelled image
def erode(image):
	return cv2.erode(image, np.ones((3, 3)))


def dilate(image):
	for _ in range(3):
		image = cv2.dilate(image, np.ones((3, 3)))
	return image


def display_intrusions(labelled):
	original = feed.get_intermediate(0)
	hist, bins = skimage.exposure.histogram(labelled)

	if np.sum(hist[1:]) / labelled.size < 0.01:
		return original

	min_area = np.mean(hist[1:])
	for area_id in bins[1:]:
		area = hist[area_id]
		if area < min_area:
			continue
		xi, yi = np.where(labelled == area_id)
		cX, cY = int(np.mean(xi)), int(np.mean(yi))
		cv2.circle(original, (cY, cX), 15, (0, 0, 255), 2)
	return original


prev_frame = None

feed = VideoFeed("IntrusionByFrameDifferencing", 0, show_src=False, show_steps=False, show_result=True)
# Turn to grayscale
feed.add_filter(lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
feed.add_filter(compute_frame_difference)
feed.add_filter(apply_thresholding)
feed.add_filter(erode)
feed.add_filter(dilate)
feed.add_filter(label_image)
feed.add_filter(display_intrusions)

while True:
	feed.process_next_frame()

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
