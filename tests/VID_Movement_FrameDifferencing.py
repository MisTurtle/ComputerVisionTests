# Detect movement using the simple but not so reliable Frame Differencing technique
# For a cleaner result, a weighed average between the previous x frames is computed to determine whether movement should be detected or not
# The following techniques I want to try, although they might be more adapted for other kind of detections (not frame to frame), are:
# - Background Subtraction
# - Adaptative Background Subtraction
# - Mixture of Gaussian


import itertools
import math

import cv2
import numpy as np

from feed import RecordingVideoFeed

warning_sequence = itertools.cycle(["/"] * 5 + ["\\"] * 5)
prev_frames = None
history_size = 100
prev_moved_percent, weights = [0 for _ in range(history_size)], [math.exp((history_size - x) / 10) for x in range(history_size)]  # Store moved percentage over x frames
feed = RecordingVideoFeed("movement_detect_test", 0, "movement_detect.mp4", show_src=True, show_result=True, show_steps=True, is_color=True)
feed.add_filter(lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # Turn grayscale
feed.add_filter(lambda frame: cv2.filter2D(frame, -1, np.ones((3, 3)) * 1/9))  # Blur to reduce noise


def perform_subtraction(gray: np.ndarray):
	if prev_frames is None:
		return gray
	return cv2.absdiff(gray, prev_frames[2])


feed.add_filter(perform_subtraction)
feed.add_filter(lambda diff: cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)[1])  # Binarize the image


def display_movement(difference: np.ndarray):
	global prev_moved_percent

	if len(difference.nonzero()) == 0:
		return feed.get_intermediate(0)

	# Compute percentage of pixels that moved
	moved = np.count_nonzero(difference == 255)
	percent_moved = moved / difference.size

	# Add it to a history
	prev_moved_percent = [percent_moved] + prev_moved_percent[:-1]
	weighed_avg = sum([prev_moved_percent[i] * weights[i] for i in range(history_size)]) / (history_size * len(weights))

	# Vary the color from green to red depending on the percentage that moved (full red = 1% or more)
	detect_after = 0.01  # percent
	red_percentage = min(1., weighed_avg / detect_after)
	fill_color = (0, 255 * (1 - red_percentage), 255 * red_percentage)

	# Fill original image with the computed color
	og = feed.get_intermediate(0)
	og[difference == 255] = fill_color

	# Add a text to indicate whether movement was detected or not
	if weighed_avg >= detect_after:
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
