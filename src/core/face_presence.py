import time


class FacePresence:
    def __init__(self, no_face_grace_seconds=0.8):
        self.no_face_grace_seconds = no_face_grace_seconds

        self.last_face_seen_at = None
        self.has_seen_face = False

    def update(self, detected_faces):
        current_time = time.monotonic()

        if len(detected_faces) > 0:
            self.last_face_seen_at = current_time
            self.has_seen_face = True

            return {"present": True, "face_count": len(detected_faces)}

        if not self.has_seen_face:
            return {"present": False, "face_count": 0}

        if self.last_face_seen_at is None:
            return {"present": False, "face_count": 0}

        missing_duration = current_time - self.last_face_seen_at

        if missing_duration < self.no_face_grace_seconds:
            return {"present": True, "face_count": 0}

        return {"present": False, "face_count": 0}


import time


class FacePresence:
    def __init__(
        self, no_face_grace_seconds=0.8, multiple_face_confirmation_seconds=0.7
    ):
        self.no_face_grace_seconds = no_face_grace_seconds

        self.multiple_face_confirmation_seconds = multiple_face_confirmation_seconds

        self.last_face_seen_at = None
        self.multiple_faces_started_at = None
        self.has_seen_face = False

    def update(self, detected_faces):
        current_time = time.monotonic()

        raw_face_count = len(detected_faces)

        if raw_face_count > 0:
            self.last_face_seen_at = current_time
            self.has_seen_face = True

        if raw_face_count > 1:
            if self.multiple_faces_started_at is None:
                self.multiple_faces_started_at = current_time

            multiple_duration = current_time - self.multiple_faces_started_at

            multiple_confirmed = (
                multiple_duration >= self.multiple_face_confirmation_seconds
            )

        else:
            self.multiple_faces_started_at = None
            multiple_confirmed = False

        if raw_face_count > 0:
            return {
                "present": True,
                "multiple_faces": multiple_confirmed,
                "face_count": (raw_face_count if multiple_confirmed else 1),
                "raw_face_count": raw_face_count,
            }

        if not self.has_seen_face:
            return {
                "present": False,
                "multiple_faces": False,
                "face_count": 0,
                "raw_face_count": 0,
            }

        if self.last_face_seen_at is None:
            return {
                "present": False,
                "multiple_faces": False,
                "face_count": 0,
                "raw_face_count": 0,
            }

        missing_duration = current_time - self.last_face_seen_at

        if missing_duration < self.no_face_grace_seconds:
            return {
                "present": True,
                "multiple_faces": False,
                "face_count": 1,
                "raw_face_count": 0,
            }

        return {
            "present": False,
            "multiple_faces": False,
            "face_count": 0,
            "raw_face_count": 0,
        }
