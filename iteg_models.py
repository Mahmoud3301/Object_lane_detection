import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os
import sys

# Paths to models
YOLO_OBJECT_PATH = "C:\\Mahmoud_Saeed\\My_projects\\project_graduation\\yolov12m.pt"
YOLO_TRAFFIC_LIGHT_PATH = "C:\\Mahmoud_Saeed\\My_projects\\project_graduation\\traffic_light\\runs\\detect\\train\\weights\\best.pt"
LANE_MODEL_PATH = "C:\\Mahmoud_Saeed\\My_projects\\project_graduation\\lane_detection_model.h5"
TRAFFIC_SIGN_MODEL_PATH = "C:\\Mahmoud_Saeed\\My_projects\\project_graduation\\traffic_sign\\traffic_sign_model.h5"
VIDEO_PATH = "C:\\Mahmoud_Saeed\\My_projects\\project_graduation\\test1.mp4"

# ========== Load models with error checks ==========
if not os.path.exists(YOLO_OBJECT_PATH):
    sys.exit(f"❌ YOLO object model not found at: {YOLO_OBJECT_PATH}")
if not os.path.exists(YOLO_TRAFFIC_LIGHT_PATH):
    sys.exit(f"❌ YOLO traffic light model not found at: {YOLO_TRAFFIC_LIGHT_PATH}")
if not os.path.exists(LANE_MODEL_PATH):
    sys.exit(f"❌ Lane detection model not found at: {LANE_MODEL_PATH}")
if not os.path.exists(TRAFFIC_SIGN_MODEL_PATH):
    sys.exit(f"❌ Traffic sign model not found at: {TRAFFIC_SIGN_MODEL_PATH}")

# Load the models
yolo_objects = YOLO(YOLO_OBJECT_PATH)
yolo_traffic_light = YOLO(YOLO_TRAFFIC_LIGHT_PATH)
lane_model = tf.keras.models.load_model(LANE_MODEL_PATH)
traffic_sign_model = tf.keras.models.load_model(TRAFFIC_SIGN_MODEL_PATH)

# Traffic light labels
traffic_light_labels = {0: "Green - go 🚦✅", 1: "Red - stop ⛔", 2: "Yellow - wait ⚠️"}

# Traffic sign labels
def get_sign_name(class_no):
    labels = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
        'Stop', 'Yield', 'No entry', 'Children crossing', 'Traffic signals', 'Turn right ahead'
    ]
    return labels[class_no] if 0 <= class_no < len(labels) else "Unknown"

# Preprocess traffic sign
def preprocess_sign(img):
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img.reshape(1, 32, 32, 1)

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    sys.exit("❌ Failed to open video.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()

    # ========== YOLO Object Detection ==========
    object_results = yolo_objects(frame, stream=False)[0]
    object_frame = object_results.plot()

    # ========== Lane Detection ==========
    lane_input = cv2.resize(frame, (320, 256))
    lane_input = np.expand_dims(lane_input, axis=0)
    lane_mask = (lane_model.predict(lane_input, verbose=0)[0] >= 0.5).astype(np.uint8) * 255
    lane_mask = cv2.resize(lane_mask, (frame.shape[1], frame.shape[0]))

    kernel = np.ones((5, 5), np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)

    lane_overlay = frame.copy()
    lane_overlay[:, :, 0] = cv2.bitwise_or(lane_overlay[:, :, 0], lane_mask)

    # ========== Traffic Light Detection ==========
    light_results = yolo_traffic_light(frame, stream=False)[0]
    light_classes = set()

    for box in light_results.boxes:
        class_id = int(box.cls.item())
        light_classes.add(class_id)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = traffic_light_labels.get(class_id, "Unknown")
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255) if class_id == 1 else (0, 255, 255)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Decision based on traffic light
    if not light_classes:
        decision = "🚦 No traffic light"
    elif 1 in light_classes:
        decision = traffic_light_labels[1]
    elif 2 in light_classes:
        decision = traffic_light_labels[2]
    else:
        decision = traffic_light_labels[0]

    cv2.putText(output_frame, f"Decision: {decision}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    # ========== Traffic Sign Classification ==========
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                preprocessed = preprocess_sign(roi)
                prediction = traffic_sign_model.predict(preprocessed, verbose=0)
                class_id = np.argmax(prediction)
                prob = np.max(prediction)
                if prob > 0.75:
                    label = get_sign_name(class_id)
                    cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(output_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Combine final view
    combined = np.hstack([
        cv2.resize(object_frame, (frame.shape[1] // 2, frame.shape[0] // 2)),
        cv2.resize(lane_overlay, (frame.shape[1] // 2, frame.shape[0] // 2))
    ])
    bottom = cv2.resize(output_frame, (frame.shape[1], frame.shape[0] // 2))
    final_display = np.vstack([combined, bottom])

    cv2.imshow("Unified Detection View", final_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








# import cv2
# import numpy as np
# import tensorflow as tf
# from ultralytics import YOLO

# # Load Models
# yolo_objects = YOLO("C:\\Mahmoud_Saeed\\My_projects\\project_graduation\\yolov12m.pt")
# yolo_traffic_light = YOLO("C:\\Mahmoud_Saeed\\My_projects\\project_graduation\\traffic_light\\runs\\detect\\train\\weights\\best.pt")
# lane_model = tf.keras.models.load_model("C:\\Mahmoud_Saeed\\My_projects\\project_graduation\\lane_detection_model.h5")
# traffic_sign_model = tf.keras.models.load_model("C:\\Mahmoud_Saeed\\My_projects\\project_graduation\\traffic_sign\\traffic_sign_model.h5")

# # Label Maps
# traffic_light_labels = {0: "Green - go ✅", 2: "Yellow - wait ⚠️", 1: "Red - stop ⛔"}
# traffic_sign_labels = [
#     'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
#     'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
#     'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons',
#     'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
#     'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
#     'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
#     'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
#     'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
#     'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left',
#     'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
# ]

# # Preprocess function for traffic sign
# def preprocess_sign_image(img):
#     try:
#         img = cv2.resize(img, (32, 32))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.equalizeHist(img)
#         img = img / 255.0
#         return img.reshape(1, 32, 32, 1)
#     except:
#         return None

# # Open video or webcam
# cap = cv2.VideoCapture("C:\\Mahmoud_Saeed\\project_graduation\\test1.mp4")  # use 0 for webcam

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Lane detection
#     lane_input = cv2.resize(frame, (320, 256))
#     lane_input = np.expand_dims(lane_input, axis=0)
#     lane_pred = lane_model.predict(lane_input)[0]
#     lane_mask = (lane_pred >= 0.5).astype(np.uint8) * 255
#     lane_mask = cv2.resize(lane_mask, (frame.shape[1], frame.shape[0]))

#     # Clean and overlay lane mask
#     kernel = np.ones((5, 5), np.uint8)
#     lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
#     lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
#     color_mask = np.zeros_like(frame)
#     color_mask[:, :, 2] = lane_mask
#     overlay = cv2.addWeighted(frame, 0.8, color_mask, 0.5, 0)

#     # Object detection (YOLO general)
#     obj_result = yolo_objects(frame, stream=False)
#     overlay = obj_result[0].plot()

#     # Traffic light detection
#     tl_result = yolo_traffic_light(frame, stream=False)
#     tl_classes = set()
#     for r in tl_result:
#         for box in r.boxes:
#             cls_id = int(box.cls.item())
#             tl_classes.add(cls_id)
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             label = traffic_light_labels.get(cls_id, "Unknown")
#             color = (0, 255, 0) if cls_id == 0 else (0, 255, 255) if cls_id == 2 else (0, 0, 255)
#             cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     # Decision logic
#     if 1 in tl_classes:
#         decision = traffic_light_labels[1]
#     elif 2 in tl_classes:
#         decision = traffic_light_labels[2]
#     elif 0 in tl_classes:
#         decision = traffic_light_labels[0]
#     else:
#         decision = "No traffic light 🚦"
#     cv2.putText(overlay, f"Decision: {decision}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

#     # Traffic sign detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
#         area = cv2.contourArea(cnt)
#         if 1000 < area < 10000:
#             x, y, w, h = cv2.boundingRect(cnt)
#             roi = frame[y:y+h, x:x+w]
#             input_img = preprocess_sign_image(roi)
#             if input_img is not None:
#                 try:
#                     pred = np.argmax(traffic_sign_model.predict(input_img), axis=1)[0]
#                     label = traffic_sign_labels[pred]
#                     cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 0), 2)
#                     cv2.putText(overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#                 except:
#                     continue

#     # Show output
#     cv2.imshow("Unified Real-Time Detection", overlay)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
