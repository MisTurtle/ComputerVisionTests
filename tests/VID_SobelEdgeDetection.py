import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from feed import VideoFeed

feed = VideoFeed("sobel_edge_detection", 0, show_result=True, show_src=True, show_steps=True)
feed.add_filter(lambda coul: cv2.cvtColor(coul, cv2.COLOR_BGR2GRAY))


def sobel_edge_detection(image: np.ndarray):
	kernel = np.array([
		[-3, 0, 3],
		[-10, 0, 10],
		[-3, 0, 3]
	])
	sobel_x = cv2.filter2D(image, cv2.CV_64F, kernel)
	abs_sobel_x = cv2.convertScaleAbs(sobel_x)
	sobel_y = cv2.filter2D(image, cv2.CV_64F, cv2.rotate(kernel, cv2.ROTATE_90_CLOCKWISE))
	abs_sobel_y = cv2.convertScaleAbs(sobel_y)
	return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)


def highlight_edges(image: np.ndarray, thresh: int, color: tuple[int, int, int]):
	src = feed.get_intermediate(0)
	src[image > thresh] = color
	return src


feed.add_filter(lambda gray: cv2.blur(gray, (3, 3)))
feed.add_filter(sobel_edge_detection)
feed.add_filter(lambda sobel: highlight_edges(sobel, 127, (255, 0, 0)))


while True:
	feed.process_next_frame()
	key = cv2.waitKey(1)
	if key & 0xFF == ord('q'):
		break
