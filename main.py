import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from insightface.app import FaceAnalysis
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTreeView,
    QTabWidget,
    QScrollArea,
)


###--------------------------------------------------------------------------###


# Import your existing functions and variables
from utils import (
    get_keypoints_and_scores,
    determine_eye_direction,
    determine_head_position,
    setup_directories,
    draw_connections,
    save_screenshot,
    draw_keypoints,
    face_app,
    EDGES,
    KEYPOINT_INDICES,
)


###--------------------------------------------------------------------------###


class InterviewProctorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_variables()
        self.setup_connections()
        self.load_models()

    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Interview Proctor")
        self.setGeometry(100, 100, 1200, 800)
        self.setup_stylesheet()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.setup_header()
        self.setup_content_layout()

    def setup_stylesheet(self):
        """Set up the application's stylesheet."""
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff; }
            QLabel { font-size: 14px; }
            QPushButton { background-color: #0078d4; color: white; border: none; padding: 5px 15px; border-radius: 3px; font-size: 14px; }
            QPushButton:hover { background-color: #005a9e; }
            QTreeView, QTabWidget::pane { background-color: #252526; border: 1px solid #3c3c3c; }
            QTabWidget::tab-bar { alignment: left; }
            QTabBar::tab { background-color: #2d2d2d; color: white; padding: 8px 20px; margin: 2px; }
            QTabBar::tab:selected { background-color: #0078d4; }
            #heading { font-size: 28px; font-weight: bold; color: #0078d4; padding: 20px 0 10px 0; }
            #explanation { font-size: 14px; color: #a0a0a0; padding: 0 20px 20px 20px; text-align: center; }
            #camera_placeholder { font-size: 18px; color: #a0a0a0; background-color: #2d2d2d; border: 1px solid #3c3c3c; }
        """
        )

    def setup_header(self):
        """Set up the header section with title and explanation."""
        self.heading = QLabel("Interview Proctor")
        self.heading.setObjectName("heading")
        self.heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.heading)

        self.explanation = QLabel(
            "This tool monitors interview participants for suspicious behavior, "
            "detecting multiple faces, eye movements, and head positions to ensure "
            "a fair and secure interview process."
        )
        self.explanation.setObjectName("explanation")
        self.explanation.setWordWrap(True)
        self.explanation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.explanation)

    def setup_content_layout(self):
        """Set up the main content layout."""
        self.content_layout = QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        self.setup_left_panel()
        self.setup_right_panel()

    def setup_left_panel(self):
        """Set up the left panel with directory view and image tabs."""
        self.left_layout = QVBoxLayout()

        self.setup_directory_view()
        self.setup_image_tabs()

        self.content_layout.addLayout(self.left_layout, 1)

    def setup_directory_view(self):
        """Set up the directory view."""
        self.directory_label = QLabel("Save Directory:")
        self.left_layout.addWidget(self.directory_label)

        self.directory_view = QTreeView()
        self.directory_model = QStandardItemModel()
        self.directory_view.setModel(self.directory_model)
        self.left_layout.addWidget(self.directory_view)

    def setup_image_tabs(self):
        """Set up the image tabs for different detection types."""
        self.image_tabs = QTabWidget()
        self.multiple_faces_tab = QScrollArea()
        self.eye_tab = QScrollArea()
        self.head_tab = QScrollArea()
        self.unusual_activity_tab = QScrollArea()  # New tab for unusual activity
        self.image_tabs.addTab(self.multiple_faces_tab, "Multiple Faces")
        self.image_tabs.addTab(self.eye_tab, "Eye")
        self.image_tabs.addTab(self.head_tab, "Head")
        self.image_tabs.addTab(self.unusual_activity_tab, "Unusual Activity")
        self.left_layout.addWidget(self.image_tabs)

    def setup_right_panel(self):
        """Set up the right panel with camera view and controls."""
        self.camera_layout = QVBoxLayout()

        self.setup_camera_label()
        self.setup_control_buttons()

        self.content_layout.addLayout(self.camera_layout, 2)

    def setup_camera_label(self):
        """Set up the camera label."""
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setObjectName("camera_placeholder")
        self.camera_label.setText(
            "Camera feed will appear here\nwhen proctoring starts"
        )
        self.camera_layout.addWidget(
            self.camera_label, alignment=Qt.AlignmentFlag.AlignCenter
        )

    def setup_control_buttons(self):
        """Set up the control buttons."""
        self.button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Proctoring")
        self.stop_button = QPushButton("Stop Proctoring")
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)
        self.camera_layout.addLayout(self.button_layout)

    def setup_variables(self):
        """Initialize variables and setup directories."""
        self.cap = None
        self.timer = QTimer()
        self.directories, self.output_dir = setup_directories()
        self.screenshot_count = 0
        self.frame_count = 0
        self.faces = []

    def setup_connections(self):
        """Set up signal-slot connections."""
        self.start_button.clicked.connect(self.start_proctoring)
        self.stop_button.clicked.connect(self.stop_proctoring)
        self.timer.timeout.connect(self.update_frame)

    def load_models(self):
        """Load TensorFlow Lite model and FaceAnalysis."""
        self.interpreter = tf.lite.Interpreter(
            model_path="model/single_pose_tflite.tflite"
        )
        self.interpreter.allocate_tensors()

    def start_proctoring(self):
        """Start the proctoring process."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)  # Update every 30 ms (approximately 33 fps)

    def stop_proctoring(self):
        """Stop the proctoring process."""
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.camera_label.setText(
                "Camera feed will appear here\nwhen proctoring starts"
            )
            self.camera_label.setStyleSheet(
                "background-color: #2d2d2d; border: 1px solid #3c3c3c;"
            )

    def update_frame(self):
        """Update the camera frame and process it."""
        ret, frame = self.cap.read()
        if ret:
            self.process_frame(frame)
            self.update_camera_display(frame)
        self.update_directory_view()

    def process_frame(self, frame):
        """Process the captured frame for face detection and pose estimation."""
        self.frame_count += 1
        if self.frame_count % 20 == 0:
            self.detect_faces(frame)

        self.estimate_pose(frame)

    def detect_faces(self, frame):
        """Detect faces in the frame."""
        faces = face_app.get(frame)
        if len(faces) > 1:
            print(f"Warning: Multiple faces detected: {len(faces)}")
            self.handle_multiple_faces(frame, faces)

    def handle_multiple_faces(self, frame, faces):
        """Handle the detection of multiple faces."""
        for face in faces:
            if "bbox" in face:
                x1, y1, x2, y2 = face["bbox"].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        self.screenshot_count = save_screenshot(
            self.directories["multiple_faces"],
            "multiple_faces",
            frame,
            self.screenshot_count,
        )
        self.update_image_tab(
            self.multiple_faces_tab, self.directories["multiple_faces"]
        )

    def estimate_pose(self, frame):
        """Estimate pose from the frame."""
        img = frame.copy()
        input_image = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
        input_image = tf.cast(input_image, dtype=tf.uint8)

        self.interpreter.set_tensor(
            self.interpreter.get_input_details()[0]["index"], np.array(input_image)
        )
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(
            self.interpreter.get_output_details()[0]["index"]
        )
        keypoints, keypoints_and_scores = get_keypoints_and_scores(
            keypoints_with_scores
        )

        if keypoints_and_scores.get("nose", 0) > 0.3:
            self.analyze_pose(frame, keypoints, keypoints_with_scores)
        elif keypoints_and_scores.get("nose", 0) < 0.1:
            self.handle_unusual_activity(frame)

    def analyze_pose(self, frame, keypoints, keypoints_with_scores):
        """Analyze the estimated pose."""
        eye_flag, eye_direction = determine_eye_direction(keypoints)
        head_flag, head_direction = determine_head_position(keypoints)
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(
            frame, keypoints_with_scores, list(KEYPOINT_INDICES.values()), 0.4
        )

        if eye_flag:
            self.handle_eye_movement(frame, eye_direction)
        if head_flag:
            self.handle_head_movement(frame, head_direction)

    def handle_eye_movement(self, frame, eye_direction):
        """Handle detected eye movement."""
        print(f"Warning: {eye_direction} detected")
        self.screenshot_count = save_screenshot(
            self.directories["eye"], "eye", frame, self.screenshot_count
        )
        self.update_image_tab(self.eye_tab, self.directories["eye"])

    def handle_head_movement(self, frame, head_direction):
        """Handle detected head movement."""
        print(f"Notice: {head_direction} head tilt detected.")
        self.screenshot_count = save_screenshot(
            self.directories["head"], "head", frame, self.screenshot_count
        )
        self.update_image_tab(self.head_tab, self.directories["head"])

    def handle_unusual_activity(self, frame):
        """Handle detected unusual activity."""
        print("Warning: Unusual Activity detected", "yellow")
        self.screenshot_count = save_screenshot(
            self.directories["unusual_activity"],
            "unusual_activity",
            frame,
            self.screenshot_count,
        )
        self.update_image_tab(
            self.unusual_activity_tab, self.directories["unusual_activity"]
        )

    def update_camera_display(self, frame):
        """Update the camera display with the processed frame."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.camera_label.setPixmap(scaled_pixmap)

    def update_directory_view(self):
        """Update the directory view."""
        self.directory_model.clear()
        root = self.directory_model.invisibleRootItem()
        self.add_directory_items(root, self.output_dir)

    def add_directory_items(self, parent, path):
        """Add directory items to the tree view."""
        for name in os.listdir(path):
            item_path = os.path.join(path, name)
            item = QStandardItem(name)
            parent.appendRow(item)
            if os.path.isdir(item_path):
                self.add_directory_items(item, item_path)

    def update_image_tab(self, tab, directory):
        """Update the image tab with recent screenshots."""
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        for filename in sorted(os.listdir(directory), reverse=True)[
            :10
        ]:  # Show last 10 images
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(directory, filename)
                pixmap = QPixmap(image_path).scaled(
                    300, 300, Qt.AspectRatioMode.KeepAspectRatio
                )
                image_label = QLabel()
                image_label.setPixmap(pixmap)
                layout.addWidget(image_label)

        tab.setWidget(content_widget)
        tab.setWidgetResizable(True)


###--------------------------------------------------------------------------###


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InterviewProctorApp()
    window.show()
    sys.exit(app.exec())
