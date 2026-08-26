import cv2


class Camera:
    def __init__(self, camera_index=0, width=640, height=480, fps=20):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.capture = None

    def start(self):
        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.capture.isOpened():
            raise RuntimeError("Unable to access webcam")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        print("[CAMERA] Camera started")

    def read(self):
        if self.capture is None:
            raise RuntimeError("Camera has not been started")

        success, frame = self.capture.read()

        if not success:
            return None

        return frame

    def stop(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        print("[CAMERA] Camera stopped")
