import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import time

class SelfDrivingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Self-Driving Car Object Detection")
        
        # Load YOLO model
        self.model = YOLO('C:\\Mahmoud_Saeed\\project_graduation\\yolov12m.pt')  # Replace with your YOLO model

        # Display all possible classes from the YOLO model in the terminal
        self.display_all_classes()

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Keep high resolution for better display
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create a label to display the video feed
        self.video_label = Label(root)
        self.video_label.pack()

        # Start video processing
        self.update_video_feed()

    def display_all_classes(self):
        # Print all model classes
        if self.model.names:
            print("YOLO Model Classes:")
            for idx, class_name in self.model.names.items():
                print(f"{idx}: {class_name}")
        else:
            print("No classes found in the YOLO model.")

    def update_video_feed(self):
        start_time = time.time()
        # Read frame from video capture
        ret, frame = self.cap.read()
        if not ret:
            return

        # Perform object detection with faster processing
        results = self.model(frame, stream=False)  # Stream=False for faster single-frame inference
        annotated_frame = results[0].plot()  # Annotate frame with detection results

        # Convert the frame to RGB for Tkinter compatibility
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update the label with the new frame
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Display frame rate in the terminal
        fps = 1.0 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")

        # Schedule the next update
        self.root.after(1, self.update_video_feed)  # Faster frame update interval

    def on_closing(self):
        # Release the video capture and close the window
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SelfDrivingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


# import cv2
# import time
# import sys
# import threading
# from ultralytics import YOLO

# class VideoStream:
#     """
#     Threaded video stream to improve frame capture rate.
#     """
#     def __init__(self, src):
#         self.cap = cv2.VideoCapture(src)
#         if not self.cap.isOpened():
#             print(f"Error: Unable to open video source {src}")
#             sys.exit(1)
#         self.ret, self.frame = self.cap.read()
#         self.stopped = False
#         self.lock = threading.Lock()
#         threading.Thread(target=self.update, daemon=True).start()

#     def update(self):
#         while not self.stopped:
#             ret, frame = self.cap.read()
#             if not ret:
#                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if needed
#                 continue
#             with self.lock:
#                 self.ret, self.frame = ret, frame

#     def read(self):
#         with self.lock:
#             return self.frame.copy() if self.ret else None  # Return None if no frame available

#     def stop(self):
#         self.stopped = True
#         self.cap.release()


# def main():
#     # Load YOLO model (update the path to your YOLO model file)
#     model_path = r'C:\Mahmoud_Saeed\project_graduation\yolov12m.pt'
#     model = YOLO(model_path)

#     # Display YOLO model classes
#     if hasattr(model, "names") and model.names:
#         print("YOLO Model Classes:")
#         for idx, class_name in model.names.items():
#             print(f"{idx}: {class_name}")
#     else:
#         print("No classes found in the YOLO model.")

#     # Initialize threaded video capture
#     video_path = r'C:\Mahmoud_Saeed\project_graduation\4.mp4'
#     video_stream = VideoStream(video_path)

#     # Retrieve video properties
#     width  = int(video_stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(video_stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps    = video_stream.cap.get(cv2.CAP_PROP_FPS) or 25
#     print(f"Input Video: {width}x{height} at {fps:.2f} FPS")

#     # Create only **one** named window
#     cv2.namedWindow("Self-Driving Car Object Detection", cv2.WINDOW_NORMAL)

#     # Process frames in a loop
#     while True:
#         start_time = time.time()
#         frame = video_stream.read()
#         if frame is None:
#             continue  # Skip if no frame is available

#         # Run object detection on the frame
#         results = model(frame, stream=False)
#         annotated_frame = results[0].plot()  # Annotate the frame

#         # Display the annotated frame in the OpenCV window
#         cv2.imshow("Self-Driving Car Object Detection", annotated_frame)

#         # Compute processing FPS (Optional)
#         elapsed = time.time() - start_time
#         processing_fps = 1.0 / elapsed if elapsed > 0 else 0
#         print(f"Processing FPS: {processing_fps:.2f}", end="\r")  # Print in the same line

#         # Exit on pressing the 'q' key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("\nExiting...")
#             break

#     # Clean up
#     video_stream.stop()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



