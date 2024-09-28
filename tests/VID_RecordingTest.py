import cv2

from feed import RecordingVideoFeed

feed = RecordingVideoFeed("rec_test", 0, "rec_test.mp4", show_src=True, show_result=True, show_steps=True, is_color=False)
feed.add_filter(lambda frame: cv2.blur(frame, (5, 5)))
feed.add_filter(lambda frame: cv2.Canny(frame, 20, 40))
feed.add_filter(lambda frame: cv2.blur(frame, (5, 5)))

while True:
	feed.process_next_frame()
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

feed.end()
