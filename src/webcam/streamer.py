"""
OpenCV VideoCapture wrapper for webcam streaming.

Manages camera resolution, FPS, and graceful release.
"""

import cv2


class WebcamStreamer:
    """Wrapper around cv2.VideoCapture."""

    def __init__(self, source=0, width=1280, height=720, fps=30):
        """
        Initialize webcam stream.

        Args:
            source: Camera index or video path.
            width: Requested frame width.
            height: Requested frame height.
            fps: Requested FPS.
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self):
        """Open the video capture."""
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return self

    def read(self):
        """Read a frame. Returns (success, frame)."""
        return self.cap.read()

    def release(self):
        """Release the capture."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
