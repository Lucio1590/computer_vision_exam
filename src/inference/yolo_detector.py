"""
YOLO Object Detector via OpenCV DNN module.

Encapsulates the full inference lifecycle:
    Load -> Transform -> Execute -> Filter -> Render
"""

import cv2
import numpy as np
from src.utils.config import YOLO_WEIGHTS_PATH, YOLO_CFG_PATH, YOLO, get_coco_classes
from src.inference.model_loader import load_yolo_network
from src.postprocessing.nms import opencv_nms
from src.preprocessing.image_pipeline import preprocess_for_yolo
from src.utils.visualization import draw_detections


class YOLODetector:
    """YOLO detector wrapper using OpenCV DNN."""

    def __init__(self):
        self.net = load_yolo_network()
        self.classes = get_coco_classes()
        self.output_layers = self._get_output_layers()
        self.conf_threshold = YOLO["conf_threshold"]
        self.nms_threshold = YOLO["nms_threshold"]

    def _get_output_layers(self):
        """Get names of output layers (yolo_82, yolo_94, yolo_106 for YOLOv3)."""
        layer_names = self.net.getLayerNames()
        return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, image: np.ndarray) -> list:
        """
        Run inference on a single image.

        Args:
            image: BGR image (H x W x 3).

        Returns:
            List of detections, each as [class_id, confidence, x, y, w, h].
        """
        height, width = image.shape[:2]
        blob = preprocess_for_yolo(image)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(detection[4]) * float(scores[class_id])

                if confidence > self.conf_threshold:
                    # YOLO returns center x, center y, width, height (normalized 0-1)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    box_w = int(detection[2] * width)
                    box_h = int(detection[3] * height)

                    # Convert to top-left corner for NMS
                    x = int(center_x - box_w / 2)
                    y = int(center_y - box_h / 2)

                    boxes.append([x, y, box_w, box_h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # Apply NMS
        indices = opencv_nms(boxes, confidences, self.conf_threshold, self.nms_threshold)

        detections = []
        for i in indices:
            x, y, w, h = boxes[i]
            detections.append([class_ids[i], confidences[i], x, y, w, h])

        return detections

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        Visualize detections on the image.

        Args:
            image: Original BGR image.
            detections: List of [class_id, confidence, x, y, w, h].

        Returns:
            Image with bounding boxes and labels.
        """
        return draw_detections(image, detections, self.classes)


def main():
    import argparse
    from pathlib import Path
    import cv2

    parser = argparse.ArgumentParser(description="YOLO Object Detector")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    detector = YOLODetector()
    detections = detector.detect(image)
    print(f"Detected {len(detections)} objects")

    annotated = detector.draw_detections(image, detections)
    out_path = output_dir / f"yolo_{image_path.name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved result to {out_path}")


if __name__ == "__main__":
    main()
