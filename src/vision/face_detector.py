import cv2
import math
import numpy as np


class FaceDetector:
    def __init__(self):
        base_path = cv2.data.haarcascades

        self.frontal_detector = cv2.CascadeClassifier(
            base_path + "haarcascade_frontalface_default.xml"
        )

        self.profile_detector = cv2.CascadeClassifier(
            base_path + "haarcascade_profileface.xml"
        )

        if self.frontal_detector.empty():
            raise RuntimeError("Unable to load frontal face detector")

        if self.profile_detector.empty():
            raise RuntimeError("Unable to load profile face detector")

        self.tilt_angles = (-15, 15)

    def _detect_frontal(self, gray):
        return self.frontal_detector.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=4, minSize=(40, 40)
        )

    def _detect_profile(self, gray):
        return self.profile_detector.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=3, minSize=(40, 40)
        )

    def _iou(self, first, second):
        x1 = max(first["x"], second["x"])
        y1 = max(first["y"], second["y"])
        x2 = min(first["x2"], second["x2"])
        y2 = min(first["y2"], second["y2"])

        intersection_width = max(0, x2 - x1)
        intersection_height = max(0, y2 - y1)

        intersection = intersection_width * intersection_height

        first_area = (first["x2"] - first["x"]) * (first["y2"] - first["y"])

        second_area = (second["x2"] - second["x"]) * (second["y2"] - second["y"])

        union = first_area + second_area - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _center_distance(self, first, second):
        first_center_x = (first["x"] + first["x2"]) / 2

        first_center_y = (first["y"] + first["y2"]) / 2

        second_center_x = (second["x"] + second["x2"]) / 2

        second_center_y = (second["y"] + second["y2"]) / 2

        return math.sqrt(
            (first_center_x - second_center_x) ** 2
            + (first_center_y - second_center_y) ** 2
        )

    def _average_size(self, face):
        width = face["x2"] - face["x"]
        height = face["y2"] - face["y"]

        return (width + height) / 2

    def _same_person(self, first, second):
        overlap = self._iou(first, second)

        if overlap >= 0.15:
            return True

        distance = self._center_distance(first, second)

        average_size = (self._average_size(first) + self._average_size(second)) / 2

        if average_size <= 0:
            return False

        normalized_distance = distance / average_size

        return normalized_distance <= 0.35

    def _merge_faces(self, first, second):
        return {
            "x": min(first["x"], second["x"]),
            "y": min(first["y"], second["y"]),
            "x2": max(first["x2"], second["x2"]),
            "y2": max(first["y2"], second["y2"]),
            "orientation": (
                first["orientation"]
                if first["orientation"] == second["orientation"]
                else "MIXED"
            ),
        }

    def _remove_duplicates(self, faces):
        merged_faces = []

        for face in faces:
            matched_index = None

            for index, existing in enumerate(merged_faces):
                if self._same_person(face, existing):
                    matched_index = index
                    break

            if matched_index is None:
                merged_faces.append(face)
            else:
                merged_faces[matched_index] = self._merge_faces(
                    merged_faces[matched_index], face
                )

        return merged_faces

    def _rotate_image(self, gray, angle):
        height, width = gray.shape[:2]

        center = (width / 2, height / 2)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            gray,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated, matrix

    def _map_rotated_face(self, x, y, w, h, rotation_matrix, frame_width, frame_height):
        inverse_matrix = cv2.invertAffineTransform(rotation_matrix)

        points = np.array(
            [[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32
        )

        points = points.reshape(-1, 1, 2)

        mapped_points = cv2.transform(points, inverse_matrix).reshape(-1, 2)

        minimum_x = max(0, int(np.min(mapped_points[:, 0])))

        minimum_y = max(0, int(np.min(mapped_points[:, 1])))

        maximum_x = min(frame_width, int(np.max(mapped_points[:, 0])))

        maximum_y = min(frame_height, int(np.max(mapped_points[:, 1])))

        if maximum_x <= minimum_x or maximum_y <= minimum_y:
            return None

        return {"x": minimum_x, "y": minimum_y, "x2": maximum_x, "y2": maximum_y}

    def _detect_tilted_faces(self, gray):
        frame_height, frame_width = gray.shape[:2]

        faces = []

        for angle in self.tilt_angles:
            rotated_gray, rotation_matrix = self._rotate_image(gray, angle)

            detections = self._detect_frontal(rotated_gray)

            for x, y, w, h in detections:
                mapped = self._map_rotated_face(
                    x, y, w, h, rotation_matrix, frame_width, frame_height
                )

                if mapped is None:
                    continue

                mapped["orientation"] = "TILT_LEFT" if angle < 0 else "TILT_RIGHT"

                faces.append(mapped)

        return self._remove_duplicates(faces)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.equalizeHist(gray)

        _, width = gray.shape

        faces = []

        frontal_faces = self._detect_frontal(gray)

        for x, y, w, h in frontal_faces:
            faces.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "x2": int(x + w),
                    "y2": int(y + h),
                    "orientation": "FRONTAL",
                }
            )

        left_profiles = self._detect_profile(gray)

        for x, y, w, h in left_profiles:
            faces.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "x2": int(x + w),
                    "y2": int(y + h),
                    "orientation": "PROFILE_LEFT",
                }
            )

        flipped_gray = cv2.flip(gray, 1)

        right_profiles = self._detect_profile(flipped_gray)

        for x, y, w, h in right_profiles:
            original_x = width - (x + w)

            faces.append(
                {
                    "x": int(original_x),
                    "y": int(y),
                    "x2": int(original_x + w),
                    "y2": int(y + h),
                    "orientation": "PROFILE_RIGHT",
                }
            )

        faces = self._remove_duplicates(faces)

        if not faces:
            faces = self._detect_tilted_faces(gray)

        return faces

    def close(self):
        pass
