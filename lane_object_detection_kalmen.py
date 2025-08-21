import cv2
import time
import sys
import threading
import numpy as np
from ultralytics import YOLO

class VideoStream:
    """
    Threaded video stream to improve frame capture rate.
    """
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

class KalmanLaneTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], 
                                                 [0, 1, 0, 1], 
                                                 [0, 0, 1, 0], 
                                                 [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.05
        self.state = np.zeros((4, 1), dtype=np.float32)
        self.kalman.statePost = self.state

    def predict(self):
        predicted = self.kalman.predict()
        return int(predicted[0]), int(predicted[1])

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32)
        self.kalman.correct(measurement)
        return self.predict()

def main():
    model_path = r'C:\Users\mahmo\OneDrive\Desktop\project_graduation\yolo11m.pt'
    model = YOLO(model_path)
    video_path = r'C:\Users\mahmo\OneDrive\Desktop\project_graduation\test1.mp4'
    video_stream = VideoStream(video_path)
    left_lane_tracker = KalmanLaneTracker()
    right_lane_tracker = KalmanLaneTracker()

    while True:
        start_time = time.time()
        frame = video_stream.read()
        height, width, _ = frame.shape

        # Object Detection
        results = model(frame, stream=False)
        annotated_frame = results[0].plot()

        # Lane Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        mask = np.zeros_like(edges)
        polygon = np.array([[(0, height), (width, height), (width//2, height//2)]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        roi_edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

        left_lane_points = []
        right_lane_points = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if slope < -0.5:
                    left_lane_points.append((x1, y1))
                    left_lane_points.append((x2, y2))
                elif slope > 0.5:
                    right_lane_points.append((x1, y1))
                    right_lane_points.append((x2, y2))

        if left_lane_points:
            avg_left_x = int(np.mean([p[0] for p in left_lane_points]))
            avg_left_y = int(np.mean([p[1] for p in left_lane_points]))
            smoothed_left_x, smoothed_left_y = left_lane_tracker.update(avg_left_x, avg_left_y)
            cv2.circle(annotated_frame, (smoothed_left_x, smoothed_left_y), 10, (0, 0, 255), -1)

        if right_lane_points:
            avg_right_x = int(np.mean([p[0] for p in right_lane_points]))
            avg_right_y = int(np.mean([p[1] for p in right_lane_points]))
            smoothed_right_x, smoothed_right_y = right_lane_tracker.update(avg_right_x, avg_right_y)
            cv2.circle(annotated_frame, (smoothed_right_x, smoothed_right_y), 10, (255, 0, 0), -1)

        cv2.imshow("Lane & Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    video_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
