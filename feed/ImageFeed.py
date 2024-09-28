from abc import ABC, abstractmethod
from typing import Union, Callable

import cv2
import numpy as np


class ImageFeed(ABC):

	reserved_names = []

	def __init__(self, name: str, **kwargs):
		assert name not in ImageFeed.reserved_names
		self._name = name
		self._src = None
		self._result = None
		self._filters = []
		self._show_source = kwargs.get('show_src', True)  # Should the frame be displayed before being filtered
		self._show_result = kwargs.get('show_result', True)  # Should the frame be displayed after being filtered
		# Should filter steps be displayed individually
		# True => Display all steps, list[int] => Display steps with a given id, False => display nothing
		self._show_steps = kwargs.get('show_steps', False)
		ImageFeed.reserved_names.append(name)

	def error(self, msg: str):
		print("\033[91m[ERR] " + self._name + ": " + msg + "\033[0m")

	def log(self, msg: str):
		print("[LOG] " + self._name + ": " + msg)

	@abstractmethod
	def process_next_frame(self):
		"""
		Starts the processing of the following frame (Fetches it from the provider and applies its filters)
		:return:
		"""
		pass

	@abstractmethod
	def _fetch_frame(self) -> Union[np.ndarray, None]:
		"""
		Fetches a frame from the provider
		"""
		pass

	def add_filter(self, fn: Callable[[np.ndarray], np.ndarray]):
		self._filters.append(fn)

	def get_filter_frame_name(self, f_id: int):
		return self._name + "_step" + str(f_id)

	def _apply_filters(self):
		if self._result is None:
			return
		for i, f in enumerate(self._filters):
			self._result = f(self._result)
			if i == len(self._filters) - 1:
				break
			if self._show_steps is True or isinstance(self._show_steps, list) and i in self._show_steps:
				cv2.imshow(self.get_filter_frame_name(i), self._result)

	def show_src(self):
		if self._src is None:
			self.error("Tried to display ImageFeed with None src")
			return
		cv2.imshow(self._name + "_src", self._src)

	def show(self):
		if self._result is None:
			self.error("Tried to display ImageFeed with None result")
			return
		cv2.imshow(self._name, self._result)

	@abstractmethod
	def end(self):
		pass
