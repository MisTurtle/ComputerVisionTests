import os
from typing import Union

import cv2

from feed import VideoFeed


class RecordingVideoFeed(VideoFeed):

	def __init__(self, name: str, in_stream: Union[str, int], out_file: str, **kwargs):
		super().__init__(name, in_stream, **kwargs)

		if not os.path.isdir("output"):
			os.mkdir("output")

		self._out_stream = cv2.VideoWriter(
			filename="output/" + out_file,
			fourcc=cv2.VideoWriter.fourcc(*kwargs.get('cc', 'mp4v')),
			fps=int(self._stream.get(cv2.CAP_PROP_FPS)),
			frameSize=(int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT))),
			isColor=kwargs.get('is_color', True)
		)

	def process_next_frame(self):
		super().process_next_frame()
		if len(self._intermediate_frames) != 0:
			self._out_stream.write(self._intermediate_frames[-1])

	def end(self):
		super().end()
		self._out_stream.release()
