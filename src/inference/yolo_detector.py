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
        # TODO: Implement full detection pipeline
        # 1. Preprocess image -> blob
        # 2. Set blob as input to network
        # 3. Forward pass through output_layers
        # 4. Parse detections:
        #    - Each output layer is a NumPy array of shape (N, 5 + num_classes)
        #    - Extract boxes, confidences, class_ids
        # 5. Apply confidence threshold
        # 6. Apply NMS (opencv_nms)
        # 7. Scale boxes back to original image dimensions
        # 8. Return filtered detections
        pass

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        Visualize detections on the image.

        Args:
            image: Original BGR image.
            detections: List of [class_id, confidence, x, y, w, h].

        Returns:
            Image with bounding boxes and labels.
        """
        # TODO: Delegate to src.utils.visualization or implement inline
        pass
