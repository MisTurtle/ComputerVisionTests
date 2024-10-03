import math
import time

import cv2
import numpy as np

from feed import VideoFeed


def sobel_edge_detection(image: np.ndarray):
	kernel = np.array([
		[-3, 0, 3],
		[-10, 0, 10],
		[-3, 0, 3]
	])
	sobel_x = cv2.filter2D(image, cv2.CV_32F, kernel)
	abs_sobel_x = cv2.convertScaleAbs(sobel_x)
	sobel_y = cv2.filter2D(image, cv2.CV_64F, cv2.rotate(kernel, cv2.ROTATE_90_CLOCKWISE))
	abs_sobel_y = cv2.convertScaleAbs(sobel_y)
	return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)


def build_mask(shape, t: float) -> cv2.UMat | np.ndarray:
	# Build a rectangle mask rotated by 45 degrees that will progressively display the edges
	mask_height = 200  # px
	lap_time = 2.5  # seconds
	t %= lap_time

	def f1(_x):
		height_c = math.sin(1.1 * math.pi * t / lap_time)
		return -_x + 2 * t * shape[1] / lap_time - mask_height * height_c

	def f2(_x):
		return -_x + 2 * t * shape[1] / lap_time

	mask = np.zeros(shape, dtype=np.uint8)

	x = np.arange(shape[1])  # x values from 0 to shape[1]-1
	y = np.arange(shape[0])[:, np.newaxis]  # y values from 0 to shape[0]-1 (as a column vector)

	# Apply f1 and f2 to the x array
	f1_values, f2_values = f1(x), f2(x)
	# Apply the mask: set values in the mask to 255 where the condition is True
	mask[(f1_values < y) & (y < f2_values)] = 255

	return mask


def draw_edges(sobel: np.ndarray, thresh: int, color: tuple[int, int, int]):
	mask = build_mask(sobel.shape, time.time() - start_time)
	cv2.imshow("Mask", mask)
	# print("Mask built in", time.time() - t, "s")
	cv2.bitwise_and(sobel, mask, sobel)
	src = feed.get_intermediate(0)
	src[sobel > thresh] = color
	return src


feed = VideoFeed("sobel_edge_detection", 0, show_result=True, show_src=False, show_steps=False)

feed.add_filter(lambda coul: cv2.cvtColor(coul, cv2.COLOR_BGR2GRAY))
feed.add_filter(lambda gray: cv2.blur(gray, (3, 3)))
feed.add_filter(sobel_edge_detection)
feed.add_filter(lambda sobel: draw_edges(sobel, 127, (30, 220, 220)))


start_time = time.time()
# start_frame = time.time()
while True:
	# start_frame = time.time()
	feed.process_next_frame()
	key = cv2.waitKey(16)
	if key & 0xFF == ord('q'):
		break
	# print("FPS: ", 1 / (time.time() - start_frame))

