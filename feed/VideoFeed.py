from typing import Union

import cv2
import numpy as np

from feed.Feed import Feed


class VideoFeed(Feed):

	def __init__(self, name: str, in_stream: Union[str, int], **kwargs):
		super().__init__(name, **kwargs)
		self._stream = cv2.VideoCapture(in_stream)
		self._enabled = True
		if not self._stream.isOpened():
			self._enabled = False
			self.error("Failed to open video stream from the following input medium -> " + str(kwargs.get('stream')))

	def process_next_frame(self):
		if not self._enabled:
			return
		self._intermediate_frames = [self._fetch_frame()]
		if self._intermediate_frames[0] is None:
			self.error("No frame could be extracted from the feed")
			return
		if self._show_source:
			self.show_src()
		self._apply_filters()
		if self._show_result:
			self.show_result()

	def _fetch_frame(self) -> Union[np.ndarray, None]:
		if not self._enabled:
			return None
		ret, cap = self._stream.read()
		if not ret:
			return None
		return cap

	def end(self):
		super().end()
		self._stream.release()
		self._stream = None
