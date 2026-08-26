from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class ObjectDetector:
    MONITORED_CLASSES = {
        0: "PERSON",
        67: "CELL_PHONE",
        73: "BOOK",
    }

    CLASS_THRESHOLDS = {
        0: 0.45,
        67: 0.35,
        73: 0.35,
    }

    def __init__(
        self,
        model_path=None,
        nms_threshold=0.45,
    ):
        if model_path is None:
            project_root = Path(__file__).resolve().parents[2]
            model_path = project_root / "models" / "yolov8n.onnx"

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise RuntimeError(f"YOLO model not found: {self.model_path}")

        session_options = ort.SessionOptions()

        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        self.input = self.session.get_inputs()[0]
        self.input_name = self.input.name

        input_shape = self.input.shape

        self.input_height = (
            int(input_shape[2]) if isinstance(input_shape[2], int) else 640
        )

        self.input_width = (
            int(input_shape[3]) if isinstance(input_shape[3], int) else 640
        )

        self.nms_threshold = nms_threshold

        print(
            "[OBJECT DETECTOR] YOLOv8n loaded "
            f"({self.input_width}x{self.input_height})"
        )

    def _letterbox(self, frame):
        frame_height, frame_width = frame.shape[:2]

        scale = min(
            self.input_width / frame_width,
            self.input_height / frame_height,
        )

        resized_width = int(frame_width * scale)
        resized_height = int(frame_height * scale)

        resized = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.full(
            (
                self.input_height,
                self.input_width,
                3,
            ),
            114,
            dtype=np.uint8,
        )

        pad_x = (self.input_width - resized_width) // 2

        pad_y = (self.input_height - resized_height) // 2

        canvas[
            pad_y : pad_y + resized_height,
            pad_x : pad_x + resized_width,
        ] = resized

        rgb = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB,
        )

        blob = rgb.astype(np.float32) / 255.0

        blob = np.transpose(
            blob,
            (2, 0, 1),
        )

        blob = np.expand_dims(
            blob,
            axis=0,
        )

        blob = np.ascontiguousarray(blob)

        return (
            blob,
            scale,
            pad_x,
            pad_y,
        )

    def _prepare_output(self, output):
        predictions = np.squeeze(output)

        if predictions.ndim != 2:
            return np.empty(
                (0, 84),
                dtype=np.float32,
            )

        if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] <= 100:
            predictions = predictions.T

        return predictions

    def _convert_box(
        self,
        center_x,
        center_y,
        width,
        height,
        scale,
        pad_x,
        pad_y,
        original_width,
        original_height,
    ):
        x1 = (center_x - width / 2 - pad_x) / scale

        y1 = (center_y - height / 2 - pad_y) / scale

        x2 = (center_x + width / 2 - pad_x) / scale

        y2 = (center_y + height / 2 - pad_y) / scale

        x1 = max(
            0,
            min(original_width - 1, x1),
        )

        y1 = max(
            0,
            min(original_height - 1, y1),
        )

        x2 = max(
            0,
            min(original_width - 1, x2),
        )

        y2 = max(
            0,
            min(original_height - 1, y2),
        )

        return (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        )

    def detect(self, frame):
        original_height, original_width = frame.shape[:2]

        (
            blob,
            scale,
            pad_x,
            pad_y,
        ) = self._letterbox(frame)

        outputs = self.session.run(
            None,
            {self.input_name: blob},
        )

        predictions = self._prepare_output(outputs[0])

        boxes = []
        scores = []
        class_ids = []

        for prediction in predictions:
            if len(prediction) < 5:
                continue

            class_scores = prediction[4:]

            class_id = int(np.argmax(class_scores))

            if class_id not in self.MONITORED_CLASSES:
                continue

            confidence = float(class_scores[class_id])

            threshold = self.CLASS_THRESHOLDS.get(
                class_id,
                0.45,
            )

            if confidence < threshold:
                continue

            center_x = float(prediction[0])

            center_y = float(prediction[1])

            width = float(prediction[2])

            height = float(prediction[3])

            (
                x1,
                y1,
                x2,
                y2,
            ) = self._convert_box(
                center_x,
                center_y,
                width,
                height,
                scale,
                pad_x,
                pad_y,
                original_width,
                original_height,
            )

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width <= 0 or box_height <= 0:
                continue

            boxes.append(
                [
                    x1,
                    y1,
                    box_width,
                    box_height,
                ]
            )

            scores.append(confidence)

            class_ids.append(class_id)

        if not boxes:
            return []

        indexes = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=0.01,
            nms_threshold=self.nms_threshold,
        )

        if len(indexes) == 0:
            return []

        detections = []

        indexes = np.array(indexes).reshape(-1)

        for index in indexes:
            x, y, width, height = boxes[index]

            class_id = class_ids[index]

            detections.append(
                {
                    "class_id": class_id,
                    "label": (self.MONITORED_CLASSES[class_id]),
                    "confidence": float(scores[index]),
                    "x": int(x),
                    "y": int(y),
                    "x2": int(x + width),
                    "y2": int(y + height),
                }
            )

        return detections

    def close(self):
        self.session = None
