from typing import Union, Callable

import cv2
import numpy as np

from feed import Feed


class PictureFeed(Feed):

	def __init__(self, name: str, image_src: str, img_mode: int, **kwargs):
		super().__init__(name, **kwargs)
		self._src = cv2.imread(image_src, img_mode)
		self._dirty = True

	def add_filter(self, fn: Callable[[np.ndarray], np.ndarray]):
		super().add_filter(fn)
		self._dirty = True

	def set_img_src(self, image_src: str, img_mode: int):
		self._src = cv2.imread(image_src, img_mode)
		self._dirty = True

	def process_next_frame(self):
		if not self._dirty:
			return
		if self._show_source:
			self.show_src()
		self._result = self._src
		self._apply_filters()
		if self._show_result:
			self.show()
		self._dirty = False

	def _fetch_frame(self) -> Union[np.ndarray, None]:
		return self._src

	def end(self):
		super().end()

