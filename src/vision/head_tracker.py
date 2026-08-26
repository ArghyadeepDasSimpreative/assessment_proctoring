import time
from datetime import datetime

import cv2
import numpy as np


class HeadTracker:
    LANDMARK_INDEXES = {
        "nose": 1,
        "chin": 152,
        "left_eye": 33,
        "right_eye": 263,
        "left_mouth": 61,
        "right_mouth": 291,
    }

    MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        calibration_seconds=2.0,
        away_threshold_seconds=3.0,
        return_confirmation_seconds=0.5,
        yaw_threshold=18.0,
        up_threshold=15.0,
        down_threshold=12.0,
    ):
        self.calibration_seconds = calibration_seconds
        self.away_threshold_seconds = away_threshold_seconds
        self.return_confirmation_seconds = return_confirmation_seconds

        self.yaw_threshold = yaw_threshold
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

        self.calibration_started_at = None
        self.calibration_samples = []

        self.calibrated = False

        self.center_yaw = 0.0
        self.center_pitch = 0.0
        self.center_roll = 0.0

        self.smoothed_yaw = None
        self.smoothed_pitch = None
        self.smoothed_roll = None

        self.smoothing_factor = 0.30

        self.away_started_at = None
        self.away_logged = False
        self.center_started_at = None

        self.last_direction = "UNKNOWN"

    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _get_image_points(
        self,
        landmarks,
        frame_width,
        frame_height,
    ):
        indexes = self.LANDMARK_INDEXES

        return np.array(
            [
                (
                    landmarks[indexes["nose"]].x * frame_width,
                    landmarks[indexes["nose"]].y * frame_height,
                ),
                (
                    landmarks[indexes["chin"]].x * frame_width,
                    landmarks[indexes["chin"]].y * frame_height,
                ),
                (
                    landmarks[indexes["left_eye"]].x * frame_width,
                    landmarks[indexes["left_eye"]].y * frame_height,
                ),
                (
                    landmarks[indexes["right_eye"]].x * frame_width,
                    landmarks[indexes["right_eye"]].y * frame_height,
                ),
                (
                    landmarks[indexes["left_mouth"]].x * frame_width,
                    landmarks[indexes["left_mouth"]].y * frame_height,
                ),
                (
                    landmarks[indexes["right_mouth"]].x * frame_width,
                    landmarks[indexes["right_mouth"]].y * frame_height,
                ),
            ],
            dtype=np.float64,
        )

    def _calculate_pose(
        self,
        landmarks,
        frame_width,
        frame_height,
    ):
        image_points = self._get_image_points(
            landmarks,
            frame_width,
            frame_height,
        )

        focal_length = float(frame_width)

        camera_matrix = np.array(
            [
                [
                    focal_length,
                    0,
                    frame_width / 2,
                ],
                [
                    0,
                    focal_length,
                    frame_height / 2,
                ],
                [
                    0,
                    0,
                    1,
                ],
            ],
            dtype=np.float64,
        )

        distortion_coefficients = np.zeros(
            (4, 1),
            dtype=np.float64,
        )

        success, rotation_vector, _ = cv2.solvePnP(
            self.MODEL_POINTS,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        angles = cv2.RQDecomp3x3(rotation_matrix)[0]

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        return yaw, pitch, roll

    def _smooth(
        self,
        yaw,
        pitch,
        roll,
    ):
        if self.smoothed_yaw is None:
            self.smoothed_yaw = yaw
            self.smoothed_pitch = pitch
            self.smoothed_roll = roll

            return yaw, pitch, roll

        alpha = self.smoothing_factor

        self.smoothed_yaw = alpha * yaw + (1 - alpha) * self.smoothed_yaw

        self.smoothed_pitch = alpha * pitch + (1 - alpha) * self.smoothed_pitch

        self.smoothed_roll = alpha * roll + (1 - alpha) * self.smoothed_roll

        return (
            self.smoothed_yaw,
            self.smoothed_pitch,
            self.smoothed_roll,
        )

    def _handle_calibration(
        self,
        yaw,
        pitch,
        roll,
    ):
        now = time.monotonic()

        if self.calibration_started_at is None:
            self.calibration_started_at = now

            print(
                "[HEAD] Calibration started. "
                "Look normally at the center of the screen."
            )

        self.calibration_samples.append(
            (
                yaw,
                pitch,
                roll,
            )
        )

        elapsed = now - self.calibration_started_at

        if elapsed < self.calibration_seconds:
            return False

        if len(self.calibration_samples) < 5:
            return False

        yaw_values = sorted(value[0] for value in self.calibration_samples)

        pitch_values = sorted(value[1] for value in self.calibration_samples)

        roll_values = sorted(value[2] for value in self.calibration_samples)

        middle = len(yaw_values) // 2

        self.center_yaw = yaw_values[middle]
        self.center_pitch = pitch_values[middle]
        self.center_roll = roll_values[middle]

        self.calibrated = True

        print(
            "[HEAD] Calibration completed "
            f"| yaw={self.center_yaw:.1f} "
            f"| pitch={self.center_pitch:.1f} "
            f"| roll={self.center_roll:.1f}"
        )

        return True

    def _calculate_direction(
        self,
        yaw,
        pitch,
        roll,
    ):
        yaw_delta = yaw - self.center_yaw

        pitch_delta = pitch - self.center_pitch

        roll_delta = roll - self.center_roll

        if abs(yaw_delta) >= self.yaw_threshold:
            if yaw_delta < 0:
                direction = "LEFT"
            else:
                direction = "RIGHT"

        elif pitch_delta >= self.down_threshold:
            direction = "DOWN"

        elif pitch_delta <= -self.up_threshold:
            direction = "UP"

        else:
            direction = "CENTER"

        return (
            direction,
            yaw_delta,
            pitch_delta,
            roll_delta,
        )

    def _update_anomaly(
        self,
        direction,
    ):
        now = time.monotonic()

        away = direction != "CENTER"

        if away:
            self.center_started_at = None

            if self.away_started_at is None:
                self.away_started_at = now

            duration = now - self.away_started_at

            if duration >= self.away_threshold_seconds and not self.away_logged:
                self.away_logged = True

                print(
                    f"[HEAD] [{self._timestamp()}] "
                    f"HEAD_AWAY > "
                    f"{self.away_threshold_seconds:.0f}s "
                    f"| Direction: {direction}"
                )

            return duration

        if self.away_logged:
            if self.center_started_at is None:
                self.center_started_at = now

            return_duration = now - self.center_started_at

            if return_duration >= self.return_confirmation_seconds:
                print(f"[HEAD] [{self._timestamp()}] " "HEAD_RETURNED_TO_SCREEN")

                self.away_logged = False
                self.away_started_at = None
                self.center_started_at = None

        else:
            self.away_started_at = None
            self.center_started_at = None

        return 0.0

    def process(
        self,
        landmarks,
        frame_shape,
    ):
        if landmarks is None:
            return {
                "valid": False,
                "direction": "UNKNOWN",
                "away": False,
                "away_duration": 0.0,
                "calibrated": self.calibrated,
            }

        frame_height = int(frame_shape[0])

        frame_width = int(frame_shape[1])

        pose = self._calculate_pose(
            landmarks,
            frame_width,
            frame_height,
        )

        if pose is None:
            return {
                "valid": False,
                "direction": "UNKNOWN",
                "away": False,
                "away_duration": 0.0,
                "calibrated": self.calibrated,
            }

        yaw, pitch, roll = self._smooth(
            pose[0],
            pose[1],
            pose[2],
        )

        if not self.calibrated:
            self._handle_calibration(
                yaw,
                pitch,
                roll,
            )

            return {
                "valid": True,
                "direction": "CALIBRATING",
                "away": False,
                "away_duration": 0.0,
                "calibrated": self.calibrated,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
            }

        (
            direction,
            yaw_delta,
            pitch_delta,
            roll_delta,
        ) = self._calculate_direction(
            yaw,
            pitch,
            roll,
        )

        self.last_direction = direction

        duration = self._update_anomaly(direction)

        return {
            "valid": True,
            "direction": direction,
            "away": direction != "CENTER",
            "away_duration": duration,
            "calibrated": True,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "yaw_delta": yaw_delta,
            "pitch_delta": pitch_delta,
            "roll_delta": roll_delta,
        }

    def reset_pending(self):
        if not self.away_logged:
            self.away_started_at = None

        self.center_started_at = None

    def recalibrate(self):
        self.calibration_started_at = None
        self.calibration_samples = []

        self.calibrated = False

        self.center_yaw = 0.0
        self.center_pitch = 0.0
        self.center_roll = 0.0

        self.smoothed_yaw = None
        self.smoothed_pitch = None
        self.smoothed_roll = None

        self.away_started_at = None
        self.away_logged = False
        self.center_started_at = None

        self.last_direction = "UNKNOWN"

        print("[HEAD] Calibration reset")

    def close(self):
        pass
