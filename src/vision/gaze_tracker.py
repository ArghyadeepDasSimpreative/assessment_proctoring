import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp


class GazeTracker:
    RIGHT_IRIS = [468, 469, 470, 471, 472]
    LEFT_IRIS = [473, 474, 475, 476, 477]

    RIGHT_EYE_HORIZONTAL = [33, 133]
    LEFT_EYE_HORIZONTAL = [362, 263]

    RIGHT_EYE_VERTICAL = [159, 145]
    LEFT_EYE_VERTICAL = [386, 374]

    def __init__(self, away_threshold_seconds=3.0, return_confirmation_seconds=0.4):
        self.away_threshold_seconds = away_threshold_seconds
        self.return_confirmation_seconds = return_confirmation_seconds

        project_root = Path(__file__).resolve().parents[2]

        model_path = project_root / "models" / "face_landmarker.task"

        if not model_path.exists():
            raise RuntimeError(f"Face Landmarker model not found: {model_path}")

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self.away_started_at = None
        self.away_logged = False
        self.center_started_at = None

        self.last_direction = "UNKNOWN"
        self.last_horizontal_ratio = None
        self.last_vertical_ratio = None

        self.last_timestamp_ms = 0

    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _average_coordinate(self, landmarks, indexes, axis):
        values = []

        for index in indexes:
            landmark = landmarks[index]

            if axis == "x":
                values.append(landmark.x)
            else:
                values.append(landmark.y)

        return sum(values) / len(values)

    def _horizontal_eye_ratio(self, landmarks, iris_indexes, eye_indexes):
        iris_x = self._average_coordinate(landmarks, iris_indexes, "x")

        first_x = landmarks[eye_indexes[0]].x

        second_x = landmarks[eye_indexes[1]].x

        minimum_x = min(first_x, second_x)

        maximum_x = max(first_x, second_x)

        width = maximum_x - minimum_x

        if width <= 0:
            return 0.5

        return (iris_x - minimum_x) / width

    def _vertical_eye_ratio(self, landmarks, iris_indexes, eye_indexes):
        iris_y = self._average_coordinate(landmarks, iris_indexes, "y")

        first_y = landmarks[eye_indexes[0]].y

        second_y = landmarks[eye_indexes[1]].y

        minimum_y = min(first_y, second_y)

        maximum_y = max(first_y, second_y)

        height = maximum_y - minimum_y

        if height <= 0.005:
            return 0.5

        return (iris_y - minimum_y) / height

    def _calculate_direction(self, landmarks):
        right_horizontal = self._horizontal_eye_ratio(
            landmarks, self.RIGHT_IRIS, self.RIGHT_EYE_HORIZONTAL
        )

        left_horizontal = self._horizontal_eye_ratio(
            landmarks, self.LEFT_IRIS, self.LEFT_EYE_HORIZONTAL
        )

        right_vertical = self._vertical_eye_ratio(
            landmarks, self.RIGHT_IRIS, self.RIGHT_EYE_VERTICAL
        )

        left_vertical = self._vertical_eye_ratio(
            landmarks, self.LEFT_IRIS, self.LEFT_EYE_VERTICAL
        )

        horizontal_ratio = (right_horizontal + left_horizontal) / 2

        vertical_ratio = (right_vertical + left_vertical) / 2

        self.last_horizontal_ratio = horizontal_ratio

        self.last_vertical_ratio = vertical_ratio

        if horizontal_ratio < 0.34:
            return "LEFT"

        if horizontal_ratio > 0.66:
            return "RIGHT"

        if vertical_ratio < 0.30:
            return "UP"

        if vertical_ratio > 0.70:
            return "DOWN"

        return "CENTER"

    def _update_anomaly_state(self, direction):
        now = time.monotonic()

        is_away = direction != "CENTER"

        if is_away:
            self.center_started_at = None

            if self.away_started_at is None:
                self.away_started_at = now

            duration = now - self.away_started_at

            if duration >= self.away_threshold_seconds and not self.away_logged:
                self.away_logged = True

                print(
                    f"[GAZE] [{self._timestamp()}] "
                    f"GAZE_AWAY detected "
                    f"for more than "
                    f"{self.away_threshold_seconds:.0f} seconds "
                    f"| Direction: {direction}"
                )

            return duration

        if self.away_logged:
            if self.center_started_at is None:
                self.center_started_at = now

            center_duration = now - self.center_started_at

            if center_duration >= self.return_confirmation_seconds:
                print(f"[GAZE] [{self._timestamp()}] " f"GAZE_RETURNED_TO_SCREEN")

                self.away_logged = False
                self.away_started_at = None
                self.center_started_at = None

        else:
            self.away_started_at = None
            self.center_started_at = None

        return 0.0

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.monotonic() * 1000)

        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1

        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            self.last_direction = "UNKNOWN"

            return {
                "direction": "UNKNOWN",
                "away": False,
                "away_duration": 0.0,
                "valid": False,
            }

        landmarks = result.face_landmarks[0]

        if len(landmarks) < 478:
            self.last_direction = "UNKNOWN"

            return {
                "direction": "UNKNOWN",
                "away": False,
                "away_duration": 0.0,
                "valid": False,
            }

        direction = self._calculate_direction(landmarks)

        self.last_direction = direction

        duration = self._update_anomaly_state(direction)

        return {
            "direction": direction,
            "away": direction != "CENTER",
            "away_duration": duration,
            "valid": True,
            "horizontal_ratio": (self.last_horizontal_ratio),
            "vertical_ratio": (self.last_vertical_ratio),
        }

    def reset_pending(self):
        if not self.away_logged:
            self.away_started_at = None

        self.center_started_at = None

    def close(self):
        self.landmarker.close()
