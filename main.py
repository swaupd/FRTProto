import sys
import os
import cv2
import numpy as np
import sqlite3
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import threading
import time
from datetime import datetime

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, QListWidget,
    QTabWidget, QMessageBox, QProgressDialog, QFrame, QSplitter,
    QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor

# ML imports
import tensorflow as tf
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity

class FaceNetEmbedding:
    """FaceNet model for generating face embeddings"""
    
    def __init__(self, model_path: str = None):
        self.input_size = 160
        self.embedding_dim = 512
        self.interpreter = None
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the FaceNet TFLite model"""
        if model_path and os.path.exists(model_path):
            try:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                print("FaceNet model loaded successfully")
            except Exception as e:
                print(f"Error loading FaceNet model: {e}")
                self.interpreter = None
        else:
            print("FaceNet model not found. Face recognition will be disabled.")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for FaceNet model"""
        # Resize to 160x160
        image = cv2.resize(image, (self.input_size, self.input_size))
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Standardize (mean=0, std=1)
        mean = np.mean(image)
        std = np.std(image)
        std = max(std, 1.0 / np.sqrt(image.size))
        image = (image - mean) / std
        
        # Add batch dimension
        return np.expand_dims(image, axis=0)
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Generate face embedding"""
        if self.interpreter is None:
            return np.random.random(self.embedding_dim)  # Fallback for demo
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(face_image)
            
            # Set input tensor
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            self.interpreter.set_tensor(input_details[0]['index'], processed_image)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            embedding = self.interpreter.get_tensor(output_details[0]['index'])
            return embedding[0]  # Remove batch dimension
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.random.random(self.embedding_dim)  # Fallback


class MediaPipeFaceDetector:
    """MediaPipe face detection wrapper"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.45
        )
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Detect faces and return cropped faces with bounding boxes"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                
                # Convert relative coordinates to absolute
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Validate bounding box
                x = max(0, x)
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                if w > 0 and h > 0:
                    cropped_face = image[y:y+h, x:x+w]
                    faces.append((cropped_face, (x, y, w, h)))
        
        return faces


class FaceDatabase:
    """SQLite database for storing face embeddings and person information"""
    
    def __init__(self, db_path: str = "face_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                embedding BLOB NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_person(self, name: str) -> int:
        """Add a new person to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("INSERT INTO persons (name) VALUES (?)", (name,))
            person_id = cursor.lastrowid
            conn.commit()
            return person_id
        except sqlite3.IntegrityError:
            # Person already exists
            cursor.execute("SELECT id FROM persons WHERE name = ?", (name,))
            person_id = cursor.fetchone()[0]
            return person_id
        finally:
            conn.close()
    
    def add_embedding(self, person_id: int, embedding: np.ndarray, image_path: str = None):
        """Add an embedding for a person"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = pickle.dumps(embedding)
        cursor.execute(
            "INSERT INTO embeddings (person_id, embedding, image_path) VALUES (?, ?, ?)",
            (person_id, embedding_blob, image_path)
        )
        
        conn.commit()
        conn.close()
    
    def get_all_persons(self) -> List[Tuple[int, str]]:
        """Get all persons from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name FROM persons ORDER BY name")
        persons = cursor.fetchall()
        
        conn.close()
        return persons
    
    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """Get all embeddings with person information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT p.id, p.name, e.embedding 
            FROM persons p 
            JOIN embeddings e ON p.id = e.person_id
        """)
        
        results = []
        for person_id, name, embedding_blob in cursor.fetchall():
            embedding = pickle.loads(embedding_blob)
            results.append((person_id, name, embedding))
        
        conn.close()
        return results
    
    def delete_person(self, person_id: int):
        """Delete a person and all their embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        
        conn.commit()
        conn.close()


class CameraThread(QThread):
    """Thread for handling camera capture"""
    
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
    
    def start_capture(self):
        self.running = True
        self.start()
    
    def stop_capture(self):
        self.running = False
        if self.cap:
            self.cap.release()
    
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            time.sleep(0.033)  # ~30 FPS
        
        if self.cap:
            self.cap.release()


class FaceRecognitionApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.face_detector = MediaPipeFaceDetector()
        self.face_net = FaceNetEmbedding("models/facenet_512.tflite")  # Update path as needed
        self.database = FaceDatabase()
        
        # Camera and recognition
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.process_frame)
        self.current_frame = None
        self.recognition_active = False
        self.known_faces = []  # Cache for known face embeddings
        
        # UI Setup
        self.setup_ui()
        self.load_known_faces()
        
        # Apply dark theme
        self.apply_dark_theme()
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Add Face Tab
        add_face_tab = self.create_add_face_tab()
        tab_widget.addTab(add_face_tab, "Add Face")
        
        # Live Detection Tab
        detection_tab = self.create_detection_tab()
        tab_widget.addTab(detection_tab, "Live Detection")
        
        # Face List Tab
        face_list_tab = self.create_face_list_tab()
        tab_widget.addTab(face_list_tab, "Face List")
        
        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(tab_widget)
        central_widget.setLayout(layout)
    
    def create_add_face_tab(self):
        """Create the add face tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Name input
        name_group = QGroupBox("Person Information")
        name_layout = QVBoxLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter person's name")
        name_layout.addWidget(QLabel("Name:"))
        name_layout.addWidget(self.name_input)
        name_group.setLayout(name_layout)
        
        # Image selection
        image_group = QGroupBox("Face Images")
        image_layout = QVBoxLayout()
        
        self.select_images_btn = QPushButton("Select Images")
        self.select_images_btn.clicked.connect(self.select_images)
        
        self.selected_images_list = QListWidget()
        self.selected_images_list.setMaximumHeight(150)
        
        image_layout.addWidget(self.select_images_btn)
        image_layout.addWidget(QLabel("Selected Images:"))
        image_layout.addWidget(self.selected_images_list)
        image_group.setLayout(image_layout)
        
        # Add person button
        self.add_person_btn = QPushButton("Add Person")
        self.add_person_btn.clicked.connect(self.add_person)
        self.add_person_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(name_group)
        layout.addWidget(image_group)
        layout.addWidget(self.add_person_btn)
        layout.addWidget(self.status_label)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def create_detection_tab(self):
        """Create the live detection tab"""
        widget = QWidget()
        layout = QHBoxLayout()
        
        # Left side - Camera feed
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Camera controls
        camera_controls = QHBoxLayout()
        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        
        self.recognition_btn = QPushButton("Start Recognition")
        self.recognition_btn.clicked.connect(self.toggle_recognition)
        self.recognition_btn.setEnabled(False)
        
        camera_controls.addWidget(self.start_camera_btn)
        camera_controls.addWidget(self.recognition_btn)
        camera_controls.addStretch()
        
        # Camera display
        self.camera_label = QLabel("Camera feed will appear here")
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid gray;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        
        left_layout.addLayout(camera_controls)
        left_layout.addWidget(self.camera_label)
        left_widget.setLayout(left_layout)
        
        # Right side - Recognition results
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        results_group = QGroupBox("Recognition Results")
        results_layout = QVBoxLayout()
        
        self.results_list = QListWidget()
        results_layout.addWidget(self.results_list)
        results_group.setLayout(results_layout)
        
        right_layout.addWidget(results_group)
        right_widget.setLayout(right_layout)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 300])
        
        layout.addWidget(splitter)
        widget.setLayout(layout)
        return widget
    
    def create_face_list_tab(self):
        """Create the face list tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_face_list)
        
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_selected_person)
        
        controls_layout.addWidget(refresh_btn)
        controls_layout.addWidget(delete_btn)
        controls_layout.addStretch()
        
        # Face list
        self.face_list_widget = QListWidget()
        self.refresh_face_list()
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.face_list_widget)
        
        widget.setLayout(layout)
        return widget
    
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        self.setPalette(dark_palette)
    
    def select_images(self):
        """Select images for face registration"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Face Images", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        self.selected_images_list.clear()
        for path in file_paths:
            self.selected_images_list.addItem(Path(path).name)
        
        # Store full paths
        self.selected_image_paths = file_paths
    
    def add_person(self):
        """Add a person to the database with their face embeddings"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name")
            return
        
        if not hasattr(self, 'selected_image_paths') or not self.selected_image_paths:
            QMessageBox.warning(self, "Warning", "Please select at least one image")
            return
        
        progress = QProgressDialog("Processing images...", "Cancel", 0, len(self.selected_image_paths), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            # Add person to database
            person_id = self.database.add_person(name)
            
            embeddings_added = 0
            for i, image_path in enumerate(self.selected_image_paths):
                if progress.wasCanceled():
                    break
                
                progress.setValue(i)
                progress.setLabelText(f"Processing {Path(image_path).name}...")
                QApplication.processEvents()
                
                # Load and process image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Detect faces
                faces = self.face_detector.detect_faces(image)
                
                for face_img, bbox in faces:
                    # Generate embedding
                    embedding = self.face_net.get_embedding(face_img)
                    
                    # Add to database
                    self.database.add_embedding(person_id, embedding, image_path)
                    embeddings_added += 1
                    break  # Use only the first face per image
            
            progress.setValue(len(self.selected_image_paths))
            
            if embeddings_added > 0:
                self.status_label.setText(f"✓ Added {name} with {embeddings_added} face embeddings")
                self.status_label.setStyleSheet("color: green;")
                
                # Clear inputs
                self.name_input.clear()
                self.selected_images_list.clear()
                if hasattr(self, 'selected_image_paths'):
                    delattr(self, 'selected_image_paths')
                
                # Reload known faces
                self.load_known_faces()
                self.refresh_face_list()
            else:
                self.status_label.setText("✗ No faces detected in the selected images")
                self.status_label.setStyleSheet("color: red;")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error adding person: {str(e)}")
        finally:
            progress.close()
    
    def load_known_faces(self):
        """Load known faces from database"""
        self.known_faces = self.database.get_all_embeddings()
        print(f"Loaded {len(self.known_faces)} known face embeddings")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_thread.running:
            self.camera_thread.start_capture()
            self.start_camera_btn.setText("Stop Camera")
            self.recognition_btn.setEnabled(True)
        else:
            self.camera_thread.stop_capture()
            self.start_camera_btn.setText("Start Camera")
            self.recognition_btn.setEnabled(False)
            self.recognition_btn.setText("Start Recognition")
            self.recognition_active = False
            self.camera_label.setText("Camera feed will appear here")
    
    def toggle_recognition(self):
        """Toggle face recognition on/off"""
        self.recognition_active = not self.recognition_active
        if self.recognition_active:
            self.recognition_btn.setText("Stop Recognition")
        else:
            self.recognition_btn.setText("Start Recognition")
    
    @pyqtSlot(np.ndarray)
    def process_frame(self, frame):
        """Process camera frame"""
        self.current_frame = frame.copy()
        display_frame = frame.copy()
        
        if self.recognition_active and len(self.known_faces) > 0:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            for face_img, (x, y, w, h) in faces:
                # Generate embedding for detected face
                face_embedding = self.face_net.get_embedding(face_img)
                
                # Find best match
                best_match, confidence = self.find_best_match(face_embedding)
                
                # Draw bounding box and label
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)  # Green if confident, orange if not
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                if best_match and confidence > 0.6:  # Threshold for recognition
                    label = f"{best_match} ({confidence:.2f})"
                    # Log recognition
                    current_time = datetime.now().strftime("%H:%M:%S")
                    self.results_list.addItem(f"[{current_time}] Detected: {best_match} (confidence: {confidence:.2f})")
                    self.results_list.scrollToBottom()
                else:
                    label = "Unknown"
                
                # Put label
                cv2.putText(display_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Convert to Qt format and display
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled_pixmap)
    
    def find_best_match(self, face_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Find the best matching person for a face embedding"""
        if not self.known_faces:
            return None, 0.0
        
        best_match = None
        best_confidence = 0.0
        
        # Compare with all known faces
        for person_id, name, known_embedding in self.known_faces:
            # Calculate cosine similarity
            similarity = cosine_similarity([face_embedding], [known_embedding])[0][0]
            
            if similarity > best_confidence:
                best_confidence = similarity
                best_match = name
        
        return best_match, best_confidence
    
    def refresh_face_list(self):
        """Refresh the face list"""
        self.face_list_widget.clear()
        persons = self.database.get_all_persons()
        
        for person_id, name in persons:
            self.face_list_widget.addItem(f"{name} (ID: {person_id})")
    
    def delete_selected_person(self):
        """Delete selected person from database"""
        current_item = self.face_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a person to delete")
            return
        
        # Extract person ID from the item text
        item_text = current_item.text()
        person_id = int(item_text.split("ID: ")[1].split(")")[0])
        person_name = item_text.split(" (ID:")[0]
        
        reply = QMessageBox.question(
            self, "Confirm Deletion", 
            f"Are you sure you want to delete {person_name}?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.database.delete_person(person_id)
            self.refresh_face_list()
            self.load_known_faces()
            QMessageBox.information(self, "Success", f"{person_name} has been deleted")
    
    def closeEvent(self, event):
        """Handle application closing"""
        if self.camera_thread.running:
            self.camera_thread.stop_capture()
            self.camera_thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better appearance
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    window = FaceRecognitionApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
