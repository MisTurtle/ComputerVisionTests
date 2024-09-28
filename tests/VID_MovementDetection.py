import itertools

import cv2
import numpy as np

from feed import RecordingVideoFeed

warning_sequence = itertools.cycle(["/"] * 5 + ["\\"] * 5)
prev_frames = None
feed = RecordingVideoFeed("movement_detect_test", 0, "movement_detect.mp4", show_src=True, show_result=True, show_steps=True, is_color=True)
feed.add_filter(lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # Turn grayscale


def perform_subtraction(gray: np.ndarray):
	if prev_frames is None:
		return gray
	# return cv2.subtract(prev_frames[1], gray)
	return cv2.subtract(gray, prev_frames[1])


feed.add_filter(perform_subtraction)
feed.add_filter(lambda diff: cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1])  # Binarize the image


def display_movement(difference: np.ndarray):
	if len(difference.nonzero()) == 0:
		return feed.get_intermediate(0)
	# Compute percentage of pixels that moved
	moved = np.count_nonzero(difference == 255)
	percent_moved = moved / difference.size
	detect_after = 0.01  # percent

	# Vary the color from green to red depending on the percentage that moved (full red = 1% or more)
	red_percentage = min(1., percent_moved / detect_after)
	fill_color = (0, 255 * (1 - red_percentage), 255 * red_percentage)

	# Fill original image with the computed color
	og = feed.get_intermediate(0)
	og[difference == 255] = fill_color

	# Add a text to indicate whether movement was detected or not
	if percent_moved >= detect_after:
		w_char = warning_sequence.__next__()
		text = w_char + " ! MOVEMENT DETECTED ! " + w_char
	else:
		text = "No movement detected"
	og = cv2.putText(og, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fill_color, 2)
	return og


feed.add_filter(display_movement)


while True:
	feed.process_next_frame()
	prev_frames = feed.extract_frames()
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

feed.end()
