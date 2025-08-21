import cv2
import time
import sys
import threading
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"Error: Unable to open video source {src}")
            sys.exit(1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()


def main():
    # Load YOLO model
    model_path = r'C:\\Mahmoud_Saeed\\project_graduation\\yolov12m.pt'
    model = YOLO(model_path)

    # Load lane detection model
    lane_model_path = r'C:\\Mahmoud_Saeed\\project_graduation\\lane_detection_model.h5'
    lane_model = tf.keras.models.load_model(lane_model_path)

    # Initialize video stream
    video_path = r'C:\\Mahmoud_Saeed\\project_graduation\\test1.mp4'
    video_stream = VideoStream(video_path)

    while True:
        start_time = time.time()
        frame = video_stream.read()
        if frame is None:
            continue

        # YOLO Object Detection
        results = model(frame, stream=False)
        annotated_frame = results[0].plot()

        # Lane Detection Segmentation
        input_frame = cv2.resize(frame, (320, 256))
        input_frame = np.expand_dims(input_frame, axis=0)
        pred_mask = lane_model.predict(input_frame)[0]
        pred_mask = (pred_mask >= 0.5).astype(np.uint8) * 255
        pred_mask = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0]))

        # Morphological processing to clean up lane mask
        kernel = np.ones((5, 5), np.uint8)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

        # Colorizing the lane mask
        mask_colored = np.zeros_like(frame)
        mask_colored[:, :, 0] = pred_mask  # Blue lanes

        # Overlay lane mask on original frame
        overlay = cv2.addWeighted(frame, 0.8, mask_colored, 0.5, 0)

        # Resize frames for display
        height, width, _ = frame.shape
        small_height = height // 2
        small_width = width // 2
        annotated_frame = cv2.resize(annotated_frame, (small_width, small_height))
        pred_mask_resized = cv2.resize(pred_mask, (small_width, small_height))
        overlay_resized = cv2.resize(overlay, (small_width, small_height))

        # Create a single display window
        combined_display = np.zeros((small_height * 2, small_width * 2, 3), dtype=np.uint8)
        combined_display[:small_height, :small_width] = cv2.cvtColor(pred_mask_resized, cv2.COLOR_GRAY2BGR)
        combined_display[:small_height, small_width:] = annotated_frame
        combined_display[small_height:, :small_width] = overlay_resized
        combined_display[small_height:, small_width:] = cv2.resize(frame, (small_width, small_height))

        cv2.imshow("Combined Display", combined_display)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    video_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
