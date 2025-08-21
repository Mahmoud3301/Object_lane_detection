# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import Label
# from PIL import Image, ImageTk
# from ultralytics import YOLO
# import time

# class LaneDetectionYOLOApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Lane and Object Detection App")

#         # Load YOLO model
#         self.model = YOLO('C:\\Users\\mahmo\\OneDrive\\Desktop\\project_graduation\\yolo11m.pt')

#         # Initialize video capture
#         self.cap = cv2.VideoCapture(0)  # Use 0 for default webcam
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#         # Create a label to display the video feed
#         self.video_label = Label(root)
#         self.video_label.pack()

#         # Start video processing
#         self.update_video_feed()

#     def region_of_interest(self, img, vertices):
#         mask = np.zeros_like(img)
#         cv2.fillPoly(mask, vertices, 255)
#         masked_img = cv2.bitwise_and(img, mask)
#         return masked_img

#     def detect_lanes(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blurred, 50, 150)

#         height, width = edges.shape
#         roi_vertices = np.array([[(50, height), (width // 2, height // 2), (width - 50, height)]], dtype=np.int32)
#         cropped_edges = self.region_of_interest(edges, roi_vertices)

#         lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=100)
        
#         lane_image = np.copy(frame)
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

#         return lane_image

#     def update_video_feed(self):
#         start_time = time.time()
        
#         ret, frame = self.cap.read()
#         if not ret:
#             return

#         frame_with_lanes = self.detect_lanes(frame)
#         results = self.model(frame_with_lanes, stream=False)  
#         annotated_frame = results[0].plot()

#         rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(rgb_frame)
#         imgtk = ImageTk.PhotoImage(image=img)

#         self.video_label.imgtk = imgtk
#         self.video_label.configure(image=imgtk)

#         fps = 1.0 / (time.time() - start_time)
#         print(f"FPS: {fps:.2f}")

#         self.root.after(1, self.update_video_feed)

#     def on_closing(self):
#         self.cap.release()
#         self.root.destroy()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = LaneDetectionYOLOApp(root)
#     root.protocol("WM_DELETE_WINDOW", app.on_closing)
#     root.mainloop()




"---------------------------------------------------------------------------------------------------------------------------------------"


import sys
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import numpy as np
import time
import threading
import concurrent.futures
from ultralytics import YOLO

##########################
# Lane Detection Section #
##########################

def detect_lane_lines(frame):
    """
    Processes the frame to detect lane lines and returns an image containing
    only the lane lines (drawn on a black background).
    """
    # 1. Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Perform Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # 3. Define a region of interest (a triangular area)
    height, width = edges.shape
    roi_corners = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.5), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_corners, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    # 4. Detect line segments using the Hough transform
    lines = cv2.HoughLinesP(cropped_edges,
                            rho=2,
                            theta=np.pi / 180,
                            threshold=100,
                            minLineLength=40,
                            maxLineGap=5)

    # 5. Average and extrapolate line segments to create one left and one right lane line
    averaged_lines = average_slope_intercept(frame, lines)

    # 6. Create an image with the lane lines drawn
    lane_lines_image = display_lines(frame, averaged_lines)

    return lane_lines_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Fit a line (y = mx + b) to the points
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            # Divide lines into left and right based on the slope sign
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    averaged_lines = []
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(image, left_fit_average)
        averaged_lines.append(left_line)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(image, right_fit_average)
        averaged_lines.append(right_line)
    return averaged_lines

def make_points(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]  # Bottom of the image
    y2 = int(y1 * 3 / 5)  # Slightly above the middle
    # Prevent division by zero
    if slope == 0:
        slope = 0.1
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]

def display_lines(image, lines):
    """
    Draws lines on a blank image (same size as input) and returns that image.
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        # Draw each line (left and right) with a thick red line
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_image

###########################
# Threaded Video Stream   #
###########################

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
            # If reached end of video, restart from beginning.
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.frame.copy()  # Return a copy to prevent race conditions

    def stop(self):
        self.stopped = True
        self.cap.release()

###########################
# YOLO Inference Function #
###########################

def run_yolo(frame, model):
    """
    Runs YOLO object detection on the frame and returns the annotated frame.
    """
    results = model(frame, stream=False)
    # The first element contains the annotated image with bounding boxes and labels.
    return results[0].plot()

###########################
# Main Application        #
###########################

def main():
    # Load the YOLO model (update the path to your YOLO model file)
    model_path = r'C:\Users\mahmo\OneDrive\Desktop\project_graduation\yolo11m.pt'
    model = YOLO(model_path)

    # (Optional) Display YOLO model classes
    if hasattr(model, "names") and model.names:
        print("YOLO Model Classes:")
        for idx, class_name in model.names.items():
            print(f"{idx}: {class_name}")
    else:
        print("No classes found in the YOLO model.")

    # Initialize threaded video capture with the video file
    video_path = r'C:\Users\mahmo\OneDrive\Desktop\project_graduation\test1.mp4'
    video_stream = VideoStream(video_path)

    # Retrieve video properties (for informational purposes)
    width  = int(video_stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video_stream.cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 200
    print(f"Input Video: {width}x{height} at {fps:.2f} FPS")

    # Create a ThreadPoolExecutor with 2 workers for concurrent processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            start_time = time.time()
            frame = video_stream.read()

            # Submit lane detection and YOLO inference concurrently
            future_lane = executor.submit(detect_lane_lines, frame)
            future_yolo = executor.submit(run_yolo, frame, model)

            # Retrieve the results once both tasks are complete
            lane_lines = future_lane.result()
            annotated_frame = future_yolo.result()

            # Combine the YOLO annotated frame and the lane detection overlay
            final_frame = cv2.addWeighted(annotated_frame, 1, lane_lines, 1, 0)

            # (Optional) Overlay FPS information on the frame.
            elapsed = time.time() - start_time
            if elapsed > 0:
                processing_fps = 1.0 / elapsed
                cv2.putText(final_frame, f"FPS: {processing_fps:.2f}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Lane & Object Detection", final_frame)

            # Exit on pressing the 'q' key.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

    video_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()








