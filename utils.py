import os
import cv2
import uuid
import numpy as np
from termcolor import colored
from insightface.app import FaceAnalysis


###--------------------------------------------------------------------------###


face_app = FaceAnalysis(
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

###--------------------------------------------------------------------------###


# Define indices for specific keypoints (example indices)
KEYPOINT_INDICES = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
}

###--------------------------------------------------------------------------###


# Define connections between keypoints of interest
EDGES = {
    (KEYPOINT_INDICES["nose"], KEYPOINT_INDICES["left_eye"]): "m",
    (KEYPOINT_INDICES["nose"], KEYPOINT_INDICES["right_eye"]): "c",
    (KEYPOINT_INDICES["left_eye"], KEYPOINT_INDICES["left_ear"]): "m",
    (KEYPOINT_INDICES["right_eye"], KEYPOINT_INDICES["right_ear"]): "c",
}


###--------------------------------------------------------------------------###


# Define function to draw connections between keypoints
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if c1 > confidence_threshold and c2 > confidence_threshold:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


###--------------------------------------------------------------------------###


# Define function to draw keypoints
def draw_keypoints(frame, keypoints, indices, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for idx in indices:
        ky, kx, kp_conf = shaped[idx]
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


###--------------------------------------------------------------------------###


def determine_eye_direction(keypoints, horizontal_threshold=0.013):
    eye_center_x = (keypoints["left_eye"][1] + keypoints["right_eye"][1]) / 2
    nose_x = keypoints["nose"][1]

    # Check horizontal directions
    if nose_x > eye_center_x + horizontal_threshold:
        return True, "Looking Right"
    elif nose_x < eye_center_x - horizontal_threshold:
        return True, "Looking Left"

    return False, "Looking Forward"


###--------------------------------------------------------------------------###


def determine_head_position(keypoints, tilt_threshold=0.09):
    ear_center_x = (keypoints["left_ear"][1] + keypoints["right_ear"][1]) / 2
    nose_x = keypoints["nose"][1]
    if nose_x > ear_center_x + tilt_threshold:
        return True, "Head Tilted Right"
    elif nose_x < ear_center_x - tilt_threshold:
        return True, "Head Tilted Left"
    return False, "Head Straight"


###--------------------------------------------------------------------------###


def get_keypoints_and_scores(keypoints_with_scores):
    keypoints_and_scores = {}

    for name, index in KEYPOINT_INDICES.items():
        # Extract score
        score = keypoints_with_scores[0, 0, index, 2]

        keypoints_and_scores[name] = float(score)

    keypoints = {
        "nose": keypoints_with_scores[0, 0, KEYPOINT_INDICES["nose"], :2],
        "left_eye": keypoints_with_scores[0, 0, KEYPOINT_INDICES["left_eye"], :2],
        "right_eye": keypoints_with_scores[0, 0, KEYPOINT_INDICES["right_eye"], :2],
        "left_ear": keypoints_with_scores[0, 0, KEYPOINT_INDICES["left_ear"], :2],
        "right_ear": keypoints_with_scores[0, 0, KEYPOINT_INDICES["right_ear"], :2],
    }
    return keypoints, keypoints_and_scores


###--------------------------------------------------------------------------###


def setup_directories(base_dir="temp"):
    output_dir = os.path.join(base_dir, str(uuid.uuid4()))
    directories = {
        "base": output_dir,
        "multiple_faces": os.path.join(output_dir, "multiple_faces"),
        "eye": os.path.join(output_dir, "eye"),
        "head": os.path.join(output_dir, "head"),
        "unusual_activity": os.path.join(output_dir, "unusual_activity"),
    }
    for path in directories.values():
        os.makedirs(path, exist_ok=True)
    return directories, output_dir


###--------------------------------------------------------------------------###


def save_screenshot(directory, category, frame, count):
    screenshot_filename = os.path.join(directory, f"{category}_{count}.png")
    cv2.imwrite(screenshot_filename, frame)
    print(colored(f"Screenshot saved as {screenshot_filename}", "green"))
    return count + 1


###--------------------------------------------------------------------------###


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
