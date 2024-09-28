from typing import Union, Callable

import cv2
import numpy as np

from feed import Feed


class PictureFeed(Feed):

	def __init__(self, name: str, img_src: str, img_mode: int, **kwargs):
		super().__init__(name, **kwargs)
		self.set_img_src(img_src, img_mode)
		self._dirty = True

	def add_filter(self, fn: Callable[[np.ndarray], np.ndarray]):
		super().add_filter(fn)
		self._dirty = True

	def set_img_src(self, image_src: str, img_mode: int):
		self._intermediate_frames = [cv2.imread(image_src, img_mode)]
		self._dirty = True

	def process_next_frame(self):
		if not self._dirty:
			return
		self._intermediate_frames = [self._intermediate_frames[0]]

		if self._show_source:
			self.show_src()
		self._apply_filters()
		if self._show_result:
			self.show_result()

		self._dirty = False

	def _fetch_frame(self) -> Union[np.ndarray, None]:
		return self._src

	def end(self):
		super().end()

