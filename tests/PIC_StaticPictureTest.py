import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from feed import PictureFeed
from matplotlib import pyplot as plt

feed = PictureFeed("static_pic_test", "input/IN_PLATE.jpeg", cv2.IMREAD_COLOR, show_result=True, show_src=True, show_steps=True)
# Color 2 grayscale
feed.add_filter(lambda og: cv2.cvtColor(og, cv2.COLOR_RGB2GRAY))


# Binarize using histogram-computed threshold
def binarize(gray: np.ndarray):
	# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
	# plt.plot(hist)
	# plt.xlim([0, 256])
	# plt.show()
	binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	return binarized


feed.add_filter(binarize)

while True:
	feed.process_next_frame()
	key = cv2.waitKey(1)
	if key & 0xFF == ord('a'):
		# Add a few filters, this makes sure that the 'dirty' property works as expected
		feed.add_filter(lambda binarized: cv2.blur(binarized, (5, 5)))
	if key & 0xFF == ord('q'):
		break
