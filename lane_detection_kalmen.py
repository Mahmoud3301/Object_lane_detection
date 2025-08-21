import cv2
import numpy as np

class KalmanLaneTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state vars (x, y, dx, dy), 2 measurement vars (x, y)
        
        # State transition matrix (predicts next position)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], 
                                                 [0, 1, 0, 1], 
                                                 [0, 0, 1, 0], 
                                                 [0, 0, 0, 1]], dtype=np.float32)

        # Measurement matrix (maps measurement to state)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0]], dtype=np.float32)

        # Process noise covariance (small value for smooth predictions)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Measurement noise covariance (measurement uncertainty)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.05

        # Initial state
        self.state = np.zeros((4, 1), dtype=np.float32)
        self.kalman.statePost = self.state

    def predict(self):
        """ Predict the next lane position. """
        predicted = self.kalman.predict()
        return int(predicted[0]), int(predicted[1])

    def update(self, x, y):
        """ Update Kalman filter with new lane detection. """
        measurement = np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32)
        self.kalman.correct(measurement)
        return self.predict()

# Initialize Kalman Filter for lane tracking
left_lane_tracker = KalmanLaneTracker()
right_lane_tracker = KalmanLaneTracker()

# Load video
cap = cv2.VideoCapture("C:\\Mahmoud_Saeed\\My_projects\\project_graduation\\test1.mp4")  # Replace with your video file

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
out = cv2.VideoWriter('lane_detection_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define region of interest
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height), (width, height), (width//2, height//2)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi_edges = cv2.bitwise_and(edges, mask)

    # Hough Line Transform
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

    left_lane_points = []
    right_lane_points = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
            if slope < -0.5:  # Left lane
                left_lane_points.append((x1, y1))
                left_lane_points.append((x2, y2))
            elif slope > 0.5:  # Right lane
                right_lane_points.append((x1, y1))
                right_lane_points.append((x2, y2))

    # Apply Kalman Filter to left and right lanes
    if left_lane_points:
        avg_left_x = int(np.mean([p[0] for p in left_lane_points]))
        avg_left_y = int(np.mean([p[1] for p in left_lane_points]))
        smoothed_left_x, smoothed_left_y = left_lane_tracker.update(avg_left_x, avg_left_y)
        cv2.circle(frame, (smoothed_left_x, smoothed_left_y), 10, (0, 0, 255), -1)  # Draw smoothed point

    if right_lane_points:
        avg_right_x = int(np.mean([p[0] for p in right_lane_points]))
        avg_right_y = int(np.mean([p[1] for p in right_lane_points]))
        smoothed_right_x, smoothed_right_y = right_lane_tracker.update(avg_right_x, avg_right_y)
        cv2.circle(frame, (smoothed_right_x, smoothed_right_y), 10, (255, 0, 0), -1)  # Draw smoothed point

    # Save and display output
    out.write(frame)
    cv2.imshow('Lane Detection with Kalman Filter', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()










