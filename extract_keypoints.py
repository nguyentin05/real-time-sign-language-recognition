"""
MediaPipe Holistic keypoint extraction for SLRNet.

Extracts pose (33×4), face (468×3), left hand (21×3) and right hand (21×3)
landmarks, concatenated into a single 1662-dimensional vector per frame.

Breakdown:
  pose      : 33 landmarks × 4 values (x, y, z, visibility) = 132
  face      : 468 landmarks × 3 values (x, y, z)            = 1404
  left_hand : 21 landmarks × 3 values (x, y, z)             = 63
  right_hand: 21 landmarks × 3 values (x, y, z)             = 63
  Total     :                                                = 1662
"""

import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    """
    Run MediaPipe Holistic detection on an image.

    Args:
        image: BGR image (numpy array from OpenCV)
        model: MediaPipe Holistic instance

    Returns:
        image  : Original BGR image (unchanged)
        results: MediaPipe detection results
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return image, results


def draw_landmarks(image, results):
    """
    Draw MediaPipe Holistic landmarks on the image.

    Args:
        image  : BGR image to draw on
        results: MediaPipe detection results
    """
    # Face mesh
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def extract_keypoints(results):
    """
    Extract and flatten all keypoints from MediaPipe Holistic results.

    Returns a 1662-dim numpy array. Missing landmarks are zero-filled.

    Args:
        results: MediaPipe detection results

    Returns:
        np.ndarray of shape (1662,)
    """
    # Pose: 33 landmarks × 4 values = 132
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    # Face: 468 landmarks × 3 values = 1404
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)

    # Left hand: 21 landmarks × 3 values = 63
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # Right hand: 21 landmarks × 3 values = 63
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])
