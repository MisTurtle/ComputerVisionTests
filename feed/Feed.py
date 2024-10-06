from abc import ABC, abstractmethod
from typing import Union, Callable

import cv2
import numpy as np


class Feed(ABC):

	reserved_names = []

	def __init__(self, name: str, **kwargs):
		assert name not in Feed.reserved_names
		self._name = name
		self._filters = []
		self._intermediate_frames = []
		self._show_source = kwargs.get('show_src', True)  # Should the frame be displayed before being filtered
		self._show_result = kwargs.get('show_result', True)  # Should the frame be displayed after being filtered
		# Should filter steps be displayed individually
		# True => Display all steps, list[int] => Display steps with a given id, False => display nothing
		self._show_steps = kwargs.get('show_steps', False)
		Feed.reserved_names.append(name)

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

	def get_intermediate(self, i: int) -> Union[np.ndarray, None]:
		if len(self._intermediate_frames) == 0:
			self.error("Tried to fetch intermediate frame from empty feed")
			return None
		return self._intermediate_frames[i % len(self._intermediate_frames)]

	def extract_frames(self):
		return self._intermediate_frames.copy()

	def add_filter(self, fn: Callable[[np.ndarray], Union[np.ndarray, cv2.UMat]]):
		self._filters.append(fn)

	def add_action(self, fn: Callable[[np.ndarray], None]):
		self._filters.append(fn)

	def get_filter_frame_name(self, f_id: int):
		return self._name + "_step" + str(f_id)

	def _apply_filters(self):
		if len(self._intermediate_frames) != 1:
			return
		for i, f in enumerate(self._filters):
			filtered = f(self._intermediate_frames[-1].copy())
			if filtered is None:  # Was an action and not a filter
				continue
			self._intermediate_frames.append(filtered)
			if i == len(self._filters) - 1:
				break
			if self._show_steps is True or isinstance(self._show_steps, list) and i in self._show_steps:
				self.show(i + 1)

	def show(self, intermediate_frame: int):
		if len(self._intermediate_frames) == 0:
			self.error("Tried to display a frame from empty feed")
			return
		intermediate_frame %= len(self._intermediate_frames)
		cv2.imshow(self._name + "_step" + str(intermediate_frame), self._intermediate_frames[intermediate_frame])

	def show_src(self):
		self.show(0)

	def show_result(self):
		self.show(-1)

	@abstractmethod
	def end(self):
		pass
