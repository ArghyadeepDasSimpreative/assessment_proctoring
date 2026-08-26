import time

import cv2

from src.vision.camera import Camera
from src.vision.face_detector import FaceDetector
from src.vision.face_landmarker import FaceLandmarker
from src.vision.gaze_tracker import GazeTracker
from src.vision.head_tracker import HeadTracker
from src.vision.object_detector import ObjectDetector
from src.core.face_presence import FacePresence
from src.core.anomaly_manager import AnomalyManager
from src.evidence.store import EvidenceStore


def main():
    camera = Camera(
        camera_index=0,
        width=640,
        height=480,
        fps=20,
    )

    face_detector = FaceDetector()

    face_presence = FacePresence(
        no_face_grace_seconds=1.5,
        multiple_face_confirmation_seconds=0.7,
    )

    face_landmarker = None

    try:
        face_landmarker = FaceLandmarker()

    except Exception as error:
        print(f"[FACE LANDMARKER INITIALIZATION ERROR] {error}")

    gaze_tracker = GazeTracker(
        away_threshold_seconds=3.0,
        return_confirmation_seconds=0.4,
    )

    head_tracker = HeadTracker(
        calibration_seconds=2.0,
        away_threshold_seconds=3.0,
        return_confirmation_seconds=0.5,
        yaw_threshold=18.0,
        up_threshold=15.0,
        down_threshold=12.0,
    )

    object_detector = None

    try:
        object_detector = ObjectDetector()

    except Exception as error:
        print(f"[OBJECT DETECTOR INITIALIZATION ERROR] {error}")

    evidence_store = EvidenceStore(
        student_id="local_test",
        exam_id="local_test",
        schedule_id="local_test",
        session_id="local_test",
    )

    anomaly_manager = AnomalyManager(evidence_store=evidence_store)

    frame_count = 0

    detected_faces = []
    object_detections = []

    shared_landmarks = None

    last_object_detection_time = 0.0
    object_detection_interval = 1.5

    face_state = {
        "present": False,
        "multiple_faces": False,
        "face_count": 0,
        "raw_face_count": 0,
    }

    gaze_state = {
        "direction": "UNKNOWN",
        "away": False,
        "away_duration": 0.0,
        "valid": False,
        "calibrated": False,
    }

    head_state = {
        "direction": "UNKNOWN",
        "away": False,
        "away_duration": 0.0,
        "valid": False,
        "calibrated": False,
    }

    try:
        camera.start()

        while True:
            frame = camera.read()

            if frame is None:
                continue

            frame = cv2.flip(
                frame,
                1,
            )

            evidence_frame = frame.copy()

            frame_count += 1

            if frame_count % 4 == 0:
                try:
                    detected_faces = face_detector.detect(frame)

                except Exception as error:
                    print(f"[FACE DETECTOR ERROR] {error}")

                    detected_faces = []

            try:
                face_state = face_presence.update(detected_faces)

            except Exception as error:
                print(f"[FACE PRESENCE ERROR] {error}")

            face_present = face_state.get(
                "present",
                False,
            )

            multiple_faces = face_state.get(
                "multiple_faces",
                False,
            )

            if (
                face_landmarker is not None
                and face_present
                and not multiple_faces
                and frame_count % 5 == 0
            ):
                try:
                    landmark_result = face_landmarker.process(frame)

                    if landmark_result.get(
                        "valid",
                        False,
                    ):
                        shared_landmarks = landmark_result.get("landmarks")

                        try:
                            gaze_state = gaze_tracker.process(
                                shared_landmarks,
                                frame.shape,
                            )

                        except Exception as error:
                            print(f"[GAZE ERROR] {error}")

                        try:
                            head_state = head_tracker.process(
                                shared_landmarks,
                                frame.shape,
                            )

                        except Exception as error:
                            print(f"[HEAD ERROR] {error}")

                    else:
                        shared_landmarks = None

                except Exception as error:
                    print(f"[FACE LANDMARKER ERROR] {error}")

                    shared_landmarks = None

            if not face_present:
                shared_landmarks = None

                try:
                    gaze_tracker.reset_pending()
                except Exception:
                    pass

                try:
                    head_tracker.reset_pending()
                except Exception:
                    pass

                gaze_state = {
                    "direction": "UNKNOWN",
                    "away": False,
                    "away_duration": 0.0,
                    "valid": False,
                    "calibrated": gaze_state.get(
                        "calibrated",
                        False,
                    ),
                }

                head_state = {
                    "direction": "UNKNOWN",
                    "away": False,
                    "away_duration": 0.0,
                    "valid": False,
                    "calibrated": head_state.get(
                        "calibrated",
                        False,
                    ),
                }

            anomaly_manager.update(
                "NO_FACE",
                active=not face_present,
                frame=evidence_frame,
                metadata={
                    "face_count": face_state.get(
                        "face_count",
                        0,
                    ),
                    "raw_face_count": face_state.get(
                        "raw_face_count",
                        0,
                    ),
                },
            )

            anomaly_manager.update(
                "MULTIPLE_FACES",
                active=multiple_faces,
                frame=evidence_frame,
                metadata={
                    "face_count": face_state.get(
                        "face_count",
                        0,
                    ),
                    "raw_face_count": face_state.get(
                        "raw_face_count",
                        0,
                    ),
                },
            )

            gaze_active = gaze_state.get(
                "valid",
                False,
            ) and gaze_state.get(
                "away",
                False,
            )

            anomaly_manager.update(
                "GAZE_AWAY",
                active=gaze_active,
                frame=evidence_frame,
                metadata={
                    "direction": gaze_state.get(
                        "direction",
                        "UNKNOWN",
                    ),
                    "away_duration": round(
                        float(
                            gaze_state.get(
                                "away_duration",
                                0.0,
                            )
                        ),
                        2,
                    ),
                    "horizontal_delta": gaze_state.get("horizontal_delta"),
                    "vertical_delta": gaze_state.get("vertical_delta"),
                },
            )

            head_away_duration = float(
                head_state.get(
                    "away_duration",
                    0.0,
                )
            )

            head_anomaly_active = (
                head_state.get(
                    "valid",
                    False,
                )
                and head_state.get(
                    "away",
                    False,
                )
                and head_away_duration >= 3.0
            )

            anomaly_manager.update(
                "HEAD_AWAY",
                active=head_anomaly_active,
                frame=evidence_frame,
                metadata={
                    "direction": head_state.get(
                        "direction",
                        "UNKNOWN",
                    ),
                    "away_duration": round(
                        head_away_duration,
                        2,
                    ),
                    "yaw": head_state.get("yaw"),
                    "pitch": head_state.get("pitch"),
                    "roll": head_state.get("roll"),
                    "yaw_delta": head_state.get("yaw_delta"),
                    "pitch_delta": head_state.get("pitch_delta"),
                },
            )

            current_time = time.monotonic()

            object_detection_ran = False

            if object_detector is not None and (
                current_time - last_object_detection_time >= object_detection_interval
            ):
                try:
                    object_detections = object_detector.detect(frame)

                except Exception as error:
                    print(f"[OBJECT DETECTOR ERROR] {error}")

                    object_detections = []

                last_object_detection_time = current_time

                object_detection_ran = True

            person_count = 0
            phone_count = 0
            book_count = 0

            for detection in object_detections:
                label = detection.get(
                    "label",
                    "UNKNOWN",
                )

                if label == "PERSON":
                    person_count += 1

                elif label == "CELL_PHONE":
                    phone_count += 1

                elif label == "BOOK":
                    book_count += 1

            if object_detection_ran:
                compact_detections = []

                for detection in object_detections:
                    compact_detections.append(
                        {
                            "label": detection.get(
                                "label",
                                "UNKNOWN",
                            ),
                            "confidence": round(
                                float(
                                    detection.get(
                                        "confidence",
                                        0.0,
                                    )
                                ),
                                3,
                            ),
                        }
                    )

                anomaly_manager.update(
                    "PHONE_DETECTED",
                    active=phone_count > 0,
                    frame=evidence_frame,
                    metadata={
                        "phone_count": phone_count,
                        "detections": compact_detections,
                    },
                )

                anomaly_manager.update(
                    "BOOK_DETECTED",
                    active=book_count > 0,
                    frame=evidence_frame,
                    metadata={
                        "book_count": book_count,
                        "detections": compact_detections,
                    },
                )

                anomaly_manager.update(
                    "EXTRA_PERSON",
                    active=person_count > 1,
                    frame=evidence_frame,
                    metadata={
                        "person_count": person_count,
                        "detections": compact_detections,
                    },
                )

            for face in detected_faces:
                try:
                    x = int(
                        face.get(
                            "x",
                            0,
                        )
                    )

                    y = int(
                        face.get(
                            "y",
                            0,
                        )
                    )

                    x2 = int(
                        face.get(
                            "x2",
                            0,
                        )
                    )

                    y2 = int(
                        face.get(
                            "y2",
                            0,
                        )
                    )

                    orientation = face.get(
                        "orientation",
                        "FACE",
                    )

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        orientation,
                        (
                            x,
                            max(
                                20,
                                y - 10,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 255, 0),
                        1,
                    )

                except Exception:
                    pass

            for detection in object_detections:
                try:
                    label = detection.get(
                        "label",
                        "UNKNOWN",
                    )

                    confidence = float(
                        detection.get(
                            "confidence",
                            0.0,
                        )
                    )

                    x = int(
                        detection.get(
                            "x",
                            0,
                        )
                    )

                    y = int(
                        detection.get(
                            "y",
                            0,
                        )
                    )

                    x2 = int(
                        detection.get(
                            "x2",
                            0,
                        )
                    )

                    y2 = int(
                        detection.get(
                            "y2",
                            0,
                        )
                    )

                    color = (255, 200, 0) if label == "PERSON" else (0, 0, 255)

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x2, y2),
                        color,
                        2,
                    )

                    cv2.putText(
                        frame,
                        (f"{label} " f"{confidence:.2f}"),
                        (
                            x,
                            max(
                                20,
                                y - 8,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        color,
                        1,
                    )

                except Exception:
                    pass

            face_count = face_state.get(
                "face_count",
                0,
            )

            if not face_present:
                face_status = "NO FACE"
                face_color = (0, 0, 255)

            elif multiple_faces:
                face_status = "MULTIPLE FACES"
                face_color = (0, 0, 255)

            else:
                face_status = "NORMAL"
                face_color = (0, 255, 0)

            gaze_direction = gaze_state.get(
                "direction",
                "UNKNOWN",
            )

            gaze_away = gaze_state.get(
                "away",
                False,
            )

            gaze_duration = float(
                gaze_state.get(
                    "away_duration",
                    0.0,
                )
            )

            if gaze_away and gaze_duration >= 3.0:
                gaze_color = (0, 0, 255)

            elif gaze_away:
                gaze_color = (0, 255, 255)

            else:
                gaze_color = (0, 255, 0)

            head_direction = head_state.get(
                "direction",
                "UNKNOWN",
            )

            if (
                head_state.get(
                    "away",
                    False,
                )
                and head_away_duration >= 3.0
            ):
                head_color = (0, 0, 255)

            elif head_state.get(
                "away",
                False,
            ):
                head_color = (0, 255, 255)

            else:
                head_color = (0, 255, 0)

            cv2.putText(
                frame,
                f"Faces: {face_count}",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                face_color,
                1,
            )

            cv2.putText(
                frame,
                f"Face: {face_status}",
                (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                face_color,
                1,
            )

            cv2.putText(
                frame,
                f"Gaze: {gaze_direction}",
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                gaze_color,
                1,
            )

            cv2.putText(
                frame,
                f"Head: {head_direction}",
                (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                head_color,
                1,
            )

            if head_state.get(
                "valid",
                False,
            ):
                yaw = float(
                    head_state.get(
                        "yaw_delta",
                        0.0,
                    )
                )

                pitch = float(
                    head_state.get(
                        "pitch_delta",
                        0.0,
                    )
                )

                cv2.putText(
                    frame,
                    (f"Yaw: {yaw:.1f} " f"Pitch: {pitch:.1f}"),
                    (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                )

            cv2.putText(
                frame,
                f"Persons: {person_count}",
                (15, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                f"Phones: {phone_count}",
                (15, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                ((0, 0, 255) if phone_count > 0 else (255, 255, 255)),
                1,
            )

            cv2.putText(
                frame,
                f"Books: {book_count}",
                (15, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                ((0, 0, 255) if book_count > 0 else (255, 255, 255)),
                1,
            )

            warning_y = 230

            if head_anomaly_active:
                cv2.putText(
                    frame,
                    "HEAD AWAY",
                    (15, warning_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 0, 255),
                    2,
                )

                warning_y += 25

            if person_count > 1:
                cv2.putText(
                    frame,
                    "EXTRA PERSON",
                    (15, warning_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 0, 255),
                    2,
                )

                warning_y += 25

            if phone_count > 0:
                cv2.putText(
                    frame,
                    "PHONE DETECTED",
                    (15, warning_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 0, 255),
                    2,
                )

                warning_y += 25

            if book_count > 0:
                cv2.putText(
                    frame,
                    "BOOK DETECTED",
                    (15, warning_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                "Press Q to stop",
                (
                    15,
                    frame.shape[0] - 15,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )

            cv2.imshow(
                "Assessment Proctor",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("[APPLICATION] Stopped")

    except Exception as error:
        print(f"[APPLICATION ERROR] {error}")

    finally:
        if object_detector is not None:
            try:
                object_detector.close()
            except Exception:
                pass

        try:
            gaze_tracker.close()
        except Exception:
            pass

        try:
            head_tracker.close()
        except Exception:
            pass

        if face_landmarker is not None:
            try:
                face_landmarker.close()
            except Exception:
                pass

        try:
            face_detector.close()
        except Exception:
            pass

        try:
            camera.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
