import time
from pathlib import Path

import cv2
import mediapipe as mp


class FaceLandmarker:
    def __init__(
        self,
        model_path=None,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        if model_path is None:
            project_root = Path(__file__).resolve().parents[2]

            model_path = project_root / "models" / "face_landmarker.task"

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise RuntimeError(f"Face Landmarker model not found: {self.model_path}")

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self.last_timestamp_ms = 0

        print("[FACE LANDMARKER] MediaPipe model loaded")

    def _get_timestamp_ms(self):
        timestamp_ms = int(time.monotonic() * 1000)

        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1

        self.last_timestamp_ms = timestamp_ms

        return timestamp_ms

    def process(self, frame):
        if frame is None:
            return {
                "valid": False,
                "landmarks": None,
                "face_count": 0,
            }

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = self._get_timestamp_ms()

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        if not result.face_landmarks:
            return {
                "valid": False,
                "landmarks": None,
                "face_count": 0,
            }

        landmarks = result.face_landmarks[0]

        if len(landmarks) < 478:
            return {
                "valid": False,
                "landmarks": None,
                "face_count": len(result.face_landmarks),
            }

        return {
            "valid": True,
            "landmarks": landmarks,
            "face_count": len(result.face_landmarks),
        }

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

        print("[FACE LANDMARKER] Closed")
