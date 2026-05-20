"""
MediaPipe Holistic keypoint extraction for SLRNet.

Extracts pose (33×4), left hand (21×3) and right hand (21×3).
To maintain compatibility with the 1662-dim model and preserve the correct 
mean/std for normalization, we FAKE the 468 face landmarks (1404 dims) 
by setting them all exactly to the nose coordinate from the pose landmarks.

Breakdown:
  pose      : 33 landmarks × 4 values (x, y, z, visibility) = 132
  face      : 468 landmarks × 3 values (x, y, z)            = 1404 (faked from nose)
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
    """
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = model.process(image_rgb)
        image_rgb.flags.writeable = True
        return image, results
    except Exception as e:
        print(f"  [WARN] MediaPipe detection failed: {e}")
        return image, None


def draw_landmarks(image, results):
    """
    Draw MediaPipe Holistic landmarks on the image (pose + hands only).
    """
    if results is None:
        return
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
    Face landmarks are faked using the nose to preserve normalization.
    """
    if results is None:
        return np.zeros(1662)
        
    # Pose: 33 landmarks × 4 values = 132
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark]).flatten()
        # Extract nose coordinates (landmark 0) to fake the face
        nose_x = pose[0]
        nose_y = pose[1]
        nose_z = pose[2]
    else:
        pose = np.zeros(132)
        nose_x, nose_y, nose_z = 0.0, 0.0, 0.0

    # Face: 468 landmarks × 3 values = 1404
    # We fake the face by using the nose coordinates 468 times.
    # This perfectly preserves the normalization center of mass for the body.
    if nose_x != 0.0 or nose_y != 0.0 or nose_z != 0.0:
        face = np.array([[nose_x, nose_y, nose_z] for _ in range(468)]).flatten()
    else:
        face = np.zeros(1404)

    # Left hand: 21 landmarks × 3 values = 63
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(63)

    # Right hand: 21 landmarks × 3 values = 63
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(63)

    return np.concatenate([pose, face, lh, rh])
