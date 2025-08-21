# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import Label
# from PIL import Image, ImageTk
# import time

# class LaneDetectionApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Real-Time Lane Detection")

#         # Initialize video capture
#         self.cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#         # Create a label to display the video feed
#         self.video_label = Label(root)
#         self.video_label.pack()

#         # Start video processing
#         self.update_video_feed()

#     def detect_lanes(self, frame):
#         # Convert the image to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#         # Canny edge detection
#         edges = cv2.Canny(blurred, 50, 150)

#         # Define a region of interest (ROI)
#         height, width = edges.shape
#         roi = np.array([[(0, height), (width, height), (width, int(height * 0.6)), (0, int(height * 0.6))]])
#         mask = np.zeros_like(edges)
#         cv2.fillPoly(mask, roi, 255)
#         masked_edges = cv2.bitwise_and(edges, mask)

#         # Hough line transform
#         lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

#         # Draw the detected lines on the original image
#         lane_image = np.copy(frame)
#         lane_lines = []
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
#                 lane_lines.append((x1, y1, x2, y2))

#         # Determine the number of lanes based on detected lines
#         num_lanes = len(lane_lines) // 2  # Assume 2 lines per lane (left and right)

#         # Annotate the number of lanes on the frame
#         cv2.putText(lane_image, f"Lanes detected: {num_lanes}", (10, 40), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         return lane_image

#     def update_video_feed(self):
#         start_time = time.time()

#         # Read frame from video capture
#         ret, frame = self.cap.read()
#         if not ret:
#             return

#         # Detect lanes in the frame
#         processed_frame = self.detect_lanes(frame)

#         # Convert the frame to RGB for Tkinter compatibility
#         rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(rgb_frame)
#         imgtk = ImageTk.PhotoImage(image=img)

#         # Update the label with the new frame
#         self.video_label.imgtk = imgtk
#         self.video_label.configure(image=imgtk)

#         # Display frame rate in the terminal
#         fps = 1.0 / (time.time() - start_time)
#         print(f"FPS: {fps:.2f}")

#         # Schedule the next update
#         self.root.after(1, self.update_video_feed)

#     def on_closing(self):
#         # Release the video capture and close the window
#         self.cap.release()
#         self.root.destroy()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = LaneDetectionApp(root)
#     root.protocol("WM_DELETE_WINDOW", app.on_closing)
#     root.mainloop()





import cv2
import numpy as np
import time

def detect_lanes(frame):
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

    # 6. Create an image to draw the lane lines
    line_image = display_lines(frame, averaged_lines)

    # 7. Overlay the line image on the original frame
    lane_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return lane_image

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
    if slope == 0:  # Prevent division by zero
        slope = 0.1
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_image

def main():
    video_path = r"C:\\Mahmoud_Saeed\\My_projects\\project_graduation\\test1.mp4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        lane_image = detect_lanes(frame)

        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0  # Prevent ZeroDivisionError

        cv2.putText(lane_image, f"FPS: {fps:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Lane Detection", lane_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()






