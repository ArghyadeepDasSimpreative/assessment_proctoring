import time
from datetime import datetime


class GazeTracker:
    RIGHT_IRIS = [468, 469, 470, 471, 472]
    LEFT_IRIS = [473, 474, 475, 476, 477]

    RIGHT_EYE_LEFT = 33
    RIGHT_EYE_RIGHT = 133
    RIGHT_EYE_TOP = 159
    RIGHT_EYE_BOTTOM = 145

    LEFT_EYE_LEFT = 362
    LEFT_EYE_RIGHT = 263
    LEFT_EYE_TOP = 386
    LEFT_EYE_BOTTOM = 374

    def __init__(
        self,
        away_threshold_seconds=3.0,
        return_confirmation_seconds=0.4,
        calibration_seconds=3.0,
        horizontal_threshold=0.08,
        vertical_threshold=0.09,
        up_threshold=0.16,
        down_threshold=None,
    ):
        self.away_threshold_seconds = away_threshold_seconds
        self.return_confirmation_seconds = return_confirmation_seconds
        self.calibration_seconds = calibration_seconds

        self.horizontal_threshold = horizontal_threshold
        self.vertical_threshold = vertical_threshold

        self.up_threshold = (
            up_threshold if up_threshold is not None else vertical_threshold
        )

        self.down_threshold = (
            down_threshold if down_threshold is not None else vertical_threshold
        )

        self.calibration_started_at = None
        self.calibration_samples = []

        self.calibrated = False

        self.center_horizontal = None
        self.center_vertical = None

        self.smoothed_horizontal = None
        self.smoothed_vertical = None

        self.smoothing_factor = 0.35

        self.away_started_at = None
        self.away_logged = False
        self.center_started_at = None

        self.last_direction = "UNKNOWN"

    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _iris_center(
        self,
        landmarks,
        indexes,
    ):
        x = sum(landmarks[index].x for index in indexes) / len(indexes)

        y = sum(landmarks[index].y for index in indexes) / len(indexes)

        return x, y

    def _eye_ratio(
        self,
        landmarks,
        iris_indexes,
        left_corner,
        right_corner,
        top_point,
        bottom_point,
    ):
        iris_x, iris_y = self._iris_center(
            landmarks,
            iris_indexes,
        )

        left_x = landmarks[left_corner].x

        right_x = landmarks[right_corner].x

        minimum_x = min(
            left_x,
            right_x,
        )

        maximum_x = max(
            left_x,
            right_x,
        )

        eye_width = maximum_x - minimum_x

        top_y = landmarks[top_point].y

        bottom_y = landmarks[bottom_point].y

        minimum_y = min(
            top_y,
            bottom_y,
        )

        maximum_y = max(
            top_y,
            bottom_y,
        )

        eye_height = maximum_y - minimum_y

        if eye_width <= 0.001:
            return None

        if eye_height <= 0.001:
            return None

        horizontal_ratio = (iris_x - minimum_x) / eye_width

        vertical_ratio = (iris_y - minimum_y) / eye_height

        return (
            horizontal_ratio,
            vertical_ratio,
        )

    def _calculate_eye_position(
        self,
        landmarks,
    ):
        if landmarks is None:
            return None

        if len(landmarks) < 478:
            return None

        right_eye = self._eye_ratio(
            landmarks,
            self.RIGHT_IRIS,
            self.RIGHT_EYE_LEFT,
            self.RIGHT_EYE_RIGHT,
            self.RIGHT_EYE_TOP,
            self.RIGHT_EYE_BOTTOM,
        )

        left_eye = self._eye_ratio(
            landmarks,
            self.LEFT_IRIS,
            self.LEFT_EYE_LEFT,
            self.LEFT_EYE_RIGHT,
            self.LEFT_EYE_TOP,
            self.LEFT_EYE_BOTTOM,
        )

        if right_eye is None or left_eye is None:
            return None

        horizontal = (right_eye[0] + left_eye[0]) / 2

        vertical = (right_eye[1] + left_eye[1]) / 2

        return (
            horizontal,
            vertical,
        )

    def _smooth(
        self,
        horizontal,
        vertical,
    ):
        if self.smoothed_horizontal is None:
            self.smoothed_horizontal = horizontal
            self.smoothed_vertical = vertical

            return (
                horizontal,
                vertical,
            )

        alpha = self.smoothing_factor

        self.smoothed_horizontal = (
            alpha * horizontal + (1 - alpha) * self.smoothed_horizontal
        )

        self.smoothed_vertical = alpha * vertical + (1 - alpha) * self.smoothed_vertical

        return (
            self.smoothed_horizontal,
            self.smoothed_vertical,
        )

    def _median(
        self,
        values,
    ):
        values = sorted(values)

        count = len(values)

        if count == 0:
            return 0.0

        middle = count // 2

        if count % 2 == 0:
            return (values[middle - 1] + values[middle]) / 2

        return values[middle]

    def _handle_calibration(
        self,
        horizontal,
        vertical,
    ):
        now = time.monotonic()

        if self.calibration_started_at is None:
            self.calibration_started_at = now

            print(
                "[GAZE] Calibration started. "
                "Look normally at the center of the laptop screen."
            )

        self.calibration_samples.append(
            (
                horizontal,
                vertical,
            )
        )

        elapsed = now - self.calibration_started_at

        if elapsed < self.calibration_seconds:
            return False

        if len(self.calibration_samples) < 5:
            return False

        horizontal_values = [sample[0] for sample in self.calibration_samples]

        vertical_values = [sample[1] for sample in self.calibration_samples]

        self.center_horizontal = self._median(horizontal_values)

        self.center_vertical = self._median(vertical_values)

        self.calibrated = True

        print(
            "[GAZE] Calibration completed "
            f"| center_x={self.center_horizontal:.3f} "
            f"| center_y={self.center_vertical:.3f}"
        )

        return True

    def _calculate_direction(
        self,
        horizontal,
        vertical,
    ):
        horizontal_delta = horizontal - self.center_horizontal

        vertical_delta = vertical - self.center_vertical

        left_strength = 0.0
        right_strength = 0.0
        up_strength = 0.0
        down_strength = 0.0

        if horizontal_delta < 0:
            left_strength = abs(horizontal_delta) / self.horizontal_threshold

        elif horizontal_delta > 0:
            right_strength = horizontal_delta / self.horizontal_threshold

        if vertical_delta < 0:
            up_strength = abs(vertical_delta) / self.up_threshold

        elif vertical_delta > 0:
            down_strength = vertical_delta / self.down_threshold

        strengths = {
            "LEFT": left_strength,
            "RIGHT": right_strength,
            "UP": up_strength,
            "DOWN": down_strength,
        }

        direction = max(
            strengths,
            key=strengths.get,
        )

        if strengths[direction] < 1.0:
            direction = "CENTER"

        return (
            direction,
            horizontal_delta,
            vertical_delta,
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
                    f"[GAZE] [{self._timestamp()}] "
                    f"GAZE_AWAY > "
                    f"{self.away_threshold_seconds:.0f}s "
                    f"| Direction: {direction}"
                )

            return duration

        if self.away_logged:
            if self.center_started_at is None:
                self.center_started_at = now

            return_duration = now - self.center_started_at

            if return_duration >= self.return_confirmation_seconds:
                print(f"[GAZE] [{self._timestamp()}] " "GAZE_RETURNED_TO_SCREEN")

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
        frame_shape=None,
    ):
        if landmarks is None:
            return {
                "direction": "UNKNOWN",
                "away": False,
                "away_duration": 0.0,
                "valid": False,
                "calibrated": self.calibrated,
            }

        if len(landmarks) < 478:
            return {
                "direction": "UNKNOWN",
                "away": False,
                "away_duration": 0.0,
                "valid": False,
                "calibrated": self.calibrated,
            }

        eye_position = self._calculate_eye_position(landmarks)

        if eye_position is None:
            return {
                "direction": "UNKNOWN",
                "away": False,
                "away_duration": 0.0,
                "valid": False,
                "calibrated": self.calibrated,
            }

        horizontal, vertical = self._smooth(
            eye_position[0],
            eye_position[1],
        )

        if not self.calibrated:
            self._handle_calibration(
                horizontal,
                vertical,
            )

            return {
                "direction": "CALIBRATING",
                "away": False,
                "away_duration": 0.0,
                "valid": True,
                "calibrated": self.calibrated,
                "horizontal_ratio": horizontal,
                "vertical_ratio": vertical,
            }

        (
            direction,
            horizontal_delta,
            vertical_delta,
        ) = self._calculate_direction(
            horizontal,
            vertical,
        )

        self.last_direction = direction

        duration = self._update_anomaly(direction)

        return {
            "direction": direction,
            "away": (direction != "CENTER"),
            "away_duration": duration,
            "valid": True,
            "calibrated": True,
            "horizontal_ratio": horizontal,
            "vertical_ratio": vertical,
            "horizontal_delta": horizontal_delta,
            "vertical_delta": vertical_delta,
        }

    def reset_pending(self):
        if not self.away_logged:
            self.away_started_at = None

        self.center_started_at = None

    def recalibrate(self):
        self.calibrated = False

        self.calibration_started_at = None
        self.calibration_samples = []

        self.center_horizontal = None
        self.center_vertical = None

        self.smoothed_horizontal = None
        self.smoothed_vertical = None

        self.away_started_at = None
        self.away_logged = False
        self.center_started_at = None

        self.last_direction = "UNKNOWN"

        print("[GAZE] Calibration reset")

    def close(self):
        pass
