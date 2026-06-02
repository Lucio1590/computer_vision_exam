"""
Geometric feature extractor for hand landmarks.

Converts 21 MediaPipe hand keypoints into a normalized feature vector
containing distances, angles, and ratios suitable for SVM classification.

MediaPipe hand landmark indices:
    0: WRIST
    1-4: THUMB (CMC, MCP, IP, TIP)
    5-8: INDEX (MCP, PIP, DIP, TIP)
    9-12: MIDDLE (MCP, PIP, DIP, TIP)
    13-16: RING (MCP, PIP, DIP, TIP)
    17-20: PINKY (MCP, PIP, DIP, TIP)
"""

import numpy as np


class FeatureExtractor:
    """Extract geometric features from 21 hand landmarks."""

    def __init__(self):
        """Initialize feature extractor."""
        pass

    def extract(self, landmarks: list) -> np.ndarray:
        """
        Extract feature vector from 21 hand landmarks.

        Features include:
            - Normalized Euclidean distances (wrist to fingertips, metacarpals)
            - Angles between joints (MCP-PIP-DIP for each finger)
            - Ratios: palm openness, finger-to-palm ratios

        Args:
            landmarks: List of 21 (x, y, z) tuples from MediaPipe.

        Returns:
            Normalized feature vector as np.ndarray.
        """
        lm = np.array(landmarks)  # shape (21, 3)

        features = []

        # 1. Distances from wrist (landmark 0) to all other landmarks
        wrist = lm[0]
        for i in range(1, 21):
            dist = np.linalg.norm(lm[i] - wrist)
            features.append(dist)

        # 2. Distances between consecutive finger joints
        finger_chains = [
            [1, 2, 3, 4],      # thumb
            [5, 6, 7, 8],      # index
            [9, 10, 11, 12],   # middle
            [13, 14, 15, 16],  # ring
            [17, 18, 19, 20],  # pinky
        ]
        for chain in finger_chains:
            for i in range(len(chain) - 1):
                dist = np.linalg.norm(lm[chain[i]] - lm[chain[i + 1]])
                features.append(dist)

        # 3. Angles at PIP joints (MCP-PIP-DIP) for each finger
        # Skip thumb (different anatomy), use index(5,6,7), middle(9,10,11), etc.
        pip_chains = [
            [5, 6, 7],      # index
            [9, 10, 11],    # middle
            [13, 14, 15],   # ring
            [17, 18, 19],   # pinky
        ]
        for a_idx, b_idx, c_idx in pip_chains:
            angle = self._angle_between(
                lm[a_idx], lm[b_idx], lm[c_idx]
            )
            features.append(angle)

        # 4. Tip-to-tip distances (spread of fingers)
        tips = [4, 8, 12, 16, 20]
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                dist = np.linalg.norm(lm[tips[i]] - lm[tips[j]])
                features.append(dist)

        # 5. Palm size reference (wrist to middle finger MCP)
        palm_size = np.linalg.norm(lm[0] - lm[9])
        if palm_size > 1e-6:
            # Normalize all accumulated distances by palm size
            # (distances are first 20 + 15 = 35 features)
            num_dist_features = 20 + 15
            for i in range(num_dist_features):
                features[i] /= palm_size

        # 6. Ratios: fingertip distance / finger length
        for idx, chain in enumerate(finger_chains):
            finger_length = sum(
                np.linalg.norm(lm[chain[j]] - lm[chain[j + 1]])
                for j in range(len(chain) - 1)
            )
            if finger_length > 1e-6:
                tip_dist = np.linalg.norm(lm[chain[-1]] - lm[chain[0]])
                features.append(tip_dist / finger_length)
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _angle_between(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Compute angle ABC (in radians) using cosine law.

        Args:
            a, b, c: 3D points. Angle is at b.

        Returns:
            Angle in radians.
        """
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0
        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))
