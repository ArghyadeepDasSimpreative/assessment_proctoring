import cv2

from src.vision.camera import Camera
from src.vision.face_detector import FaceDetector
from src.vision.gaze_tracker import GazeTracker
from src.core.face_presence import FacePresence


def main():
    camera = Camera(camera_index=0, width=640, height=480, fps=20)

    face_detector = FaceDetector()

    face_presence = FacePresence(
        no_face_grace_seconds=1.5, multiple_face_confirmation_seconds=0.7
    )

    gaze_tracker = GazeTracker(
        away_threshold_seconds=3.0, return_confirmation_seconds=0.4
    )

    frame_count = 0

    detected_faces = []

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
    }

    try:
        camera.start()

        while True:
            frame = camera.read()

            if frame is None:
                continue

            frame = cv2.flip(frame, 1)

            frame_count += 1

            if frame_count % 4 == 0:
                try:
                    detected_faces = face_detector.detect(frame)

                except Exception as error:
                    print(f"[FACE DETECTOR ERROR] " f"{error}")

                    detected_faces = []

            try:
                face_state = face_presence.update(detected_faces)

            except Exception as error:
                print(f"[FACE PRESENCE ERROR] " f"{error}")

            if frame_count % 5 == 0 and face_state.get("present", False):
                try:
                    gaze_state = gaze_tracker.process(frame)

                except Exception as error:
                    print(f"[GAZE ERROR] {error}")

            if not face_state.get("present", False):
                gaze_tracker.reset_pending()

                gaze_state = {
                    "direction": "UNKNOWN",
                    "away": False,
                    "away_duration": 0.0,
                    "valid": False,
                }

            for face in detected_faces:
                x = int(face.get("x", 0))

                y = int(face.get("y", 0))

                x2 = int(face.get("x2", 0))

                y2 = int(face.get("y2", 0))

                orientation = face.get("orientation", "FACE")

                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    orientation,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )

            present = face_state.get("present", False)

            multiple_faces = face_state.get("multiple_faces", False)

            face_count = face_state.get("face_count", 0)

            if not present:
                face_status = "NO FACE"
                face_color = (0, 0, 255)

            elif multiple_faces:
                face_status = "MULTIPLE FACES"

                face_color = (0, 0, 255)

            else:
                face_status = "NORMAL"
                face_color = (0, 255, 0)

            gaze_direction = gaze_state.get("direction", "UNKNOWN")

            gaze_away = gaze_state.get("away", False)

            gaze_duration = gaze_state.get("away_duration", 0.0)

            if gaze_away and gaze_duration >= 3.0:
                gaze_color = (0, 0, 255)
            elif gaze_away:
                gaze_color = (0, 255, 255)
            else:
                gaze_color = (0, 255, 0)

            cv2.putText(
                frame,
                f"Faces: {face_count}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                face_color,
                2,
            )

            cv2.putText(
                frame,
                f"Face Status: {face_status}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                face_color,
                2,
            )

            cv2.putText(
                frame,
                f"Gaze: {gaze_direction}",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                gaze_color,
                2,
            )

            if gaze_away:
                cv2.putText(
                    frame,
                    ("Gaze Away: " f"{gaze_duration:.1f}s"),
                    (15, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    gaze_color,
                    2,
                )

            cv2.putText(
                frame,
                "Press Q to stop",
                (15, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Assessment Proctor", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    except KeyboardInterrupt:
        print("[APPLICATION] Stopped")

    except Exception as error:
        print(f"[APPLICATION ERROR] " f"{error}")

    finally:
        try:
            gaze_tracker.close()
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
