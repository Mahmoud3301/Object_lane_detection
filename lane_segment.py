# import cv2
# import numpy as np
# import tensorflow as tf

# # تحميل النموذج المدرب
# model = tf.keras.models.load_model('C:\\Users\\mahmo\\OneDrive\\Desktop\\project_graduation\\lane_detection_model.h5')

# # تشغيل كاميرا الويب
# cap = cv2.VideoCapture(0)  # استخدم 0 للكاميرا الافتراضية

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # إذا لم يتم التقاط إطار، أخرج من الحلقة

#     # تغيير حجم الصورة لتتناسب مع مدخلات النموذج
#     input_frame = cv2.resize(frame, (320, 256))
    
#     # تحويل الصورة إلى صيغة مناسبة للنموذج
#     input_frame = np.expand_dims(input_frame, axis=0)  # إضافة البعد الدفعي
    
#     # التنبؤ باستخدام النموذج
#     pred_mask = model.predict(input_frame)[0]
    
#     # تحويل المخرجات إلى قناع ثنائي
#     pred_mask = (pred_mask >= 0.5).astype(np.uint8) * 255  # تحويل إلى 0 و 255
    
#     # تغيير الحجم ليطابق حجم الإطار الأصلي
#     pred_mask = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0]))

#     # تحويل القناع إلى صورة ملونة (أحمر للمسارات)
#     mask_colored = np.zeros_like(frame)  # إنشاء صورة سوداء بحجم الفيديو
#     mask_colored[:, :, 0] = pred_mask  # وضع القناع في القناة الحمراء (يمكن تغييره للأخضر)

#     # دمج القناع مع الفيديو الأصلي
#     overlay = cv2.addWeighted(frame, 0.8, mask_colored, 0.5, 0)  # شفافية 50% للقناع
    
#     # عرض الصورة الأصلية
#     cv2.imshow('Real-Time Lane Detection', frame)

#     # عرض القناع المتنبأ به فقط
#     cv2.imshow('Lane Segmentation', pred_mask)

#     # عرض القناع فوق الصورة الأصلية
#     cv2.imshow('Lane Overlay', overlay)

#     # اضغط 'q' للخروج
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # تحرير الموارد
# cap.release()
# cv2.destroyAllWindows()






# import cv2
# import numpy as np
# import tensorflow as tf

# # تحميل النموذج المدرب
# model = tf.keras.models.load_model('C:\\Users\\mahmo\\OneDrive\\Desktop\\project_graduation\\lane_detection_model.h5')

# # تشغيل الفيديو
# video_path = "C:\\Users\\mahmo\\OneDrive\\Desktop\\project_graduation\\test1.mp4"  # استبدل بمسار الفيديو
# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # إذا انتهى الفيديو، اخرج من الحلقة

#     # تغيير حجم الصورة لتتناسب مع مدخلات النموذج
#     input_frame = cv2.resize(frame, (320, 256))
#     input_frame = np.expand_dims(input_frame, axis=0)  # إضافة البعد الدفعي

#     # التنبؤ باستخدام النموذج
#     pred_mask = model.predict(input_frame)[0]
#     pred_mask = (pred_mask >= 0.5).astype(np.uint8) * 255  # تحويل إلى 0 و 255
#     pred_mask = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0]))

#     # إزالة الضوضاء باستخدام العمليات المورفولوجية
#     kernel = np.ones((5, 5), np.uint8)
#     pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)  # فتح لإزالة الضوضاء الصغيرة
#     pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)  # غلق لدمج الأجزاء المتقطعة

#     # إنشاء قناع ملون للأشرطة الزرقاء فقط في المنتصف
#     mask_colored = np.zeros_like(frame)
#     mask_colored[:, :, 0] = pred_mask  # اللون الأزرق للأشرطة

#     # حجب الجوانب غير المرغوب فيها
#     height, width = frame.shape[:2]
#     roi_mask = np.zeros((height, width), np.uint8)
#     cv2.rectangle(roi_mask, (int(width * 0.2), 0), (int(width * 0.8), height), 255, -1)
#     pred_mask = cv2.bitwise_and(pred_mask, pred_mask, mask=roi_mask)
#     mask_colored = cv2.bitwise_and(mask_colored, mask_colored, mask=roi_mask)

#     # دمج القناع مع الفيديو الأصلي
#     overlay = cv2.addWeighted(frame, 0.8, mask_colored, 0.5, 0)

#     # عرض الفيديو
#     cv2.imshow('Real-Time Lane Detection', frame)
#     cv2.imshow('Lane Segmentation', pred_mask)
#     cv2.imshow('Lane Overlay', overlay)

#     # اضغط 'q' للخروج
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # تحرير الموارد
# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import tensorflow as tf

# ✅ Load the trained model (fixes the compilation warning)
model_path = "C:\\Users\\mahmo\\OneDrive\\Desktop\\project_graduation\\lane_detection_model.h5"
model = tf.keras.models.load_model(model_path, compile=False)

# ✅ Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("✅ Using GPU:", physical_devices[0])
else:
    print("⚠️ Using CPU. Consider installing tensorflow-gpu for better performance.")

# ✅ Load the video
video_path = "C:\\Users\\mahmo\\OneDrive\\Desktop\\project_graduation\\test1.mp4"
cap = cv2.VideoCapture(video_path)

# ✅ Reduce video frame processing time
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
target_size = (320, 256)  # Adjust to match model input

# ✅ Define morphological operations kernel
kernel = np.ones((5, 5), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends

    # ✅ Resize frame for model input
    input_frame = cv2.resize(frame, target_size)
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension

    # ✅ Predict lane segmentation mask
    pred_mask = model.predict(input_frame, verbose=0)[0]
    pred_mask = (pred_mask >= 0.5).astype(np.uint8) * 255  # Binarize mask
    pred_mask = cv2.resize(pred_mask, (frame_width, frame_height))  # Resize to original frame size

    # ✅ Apply morphological operations (reduce noise, improve mask clarity)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

    # ✅ Create a colored mask overlay (blue lanes)
    mask_colored = np.zeros_like(frame)
    mask_colored[:, :, 0] = pred_mask  # Assigning to blue channel

    # ✅ Define Region of Interest (ROI) mask to remove unwanted areas
    roi_mask = np.zeros((frame_height, frame_width), np.uint8)
    cv2.rectangle(roi_mask, (int(frame_width * 0.2), 0), (int(frame_width * 0.8), frame_height), 255, -1)
    pred_mask = cv2.bitwise_and(pred_mask, pred_mask, mask=roi_mask)
    mask_colored = cv2.bitwise_and(mask_colored, mask_colored, mask=roi_mask)

    # ✅ Overlay mask on original frame
    overlay = cv2.addWeighted(frame, 0.8, mask_colored, 0.5, 0)

    # ✅ Display results
    cv2.imshow('Real-Time Lane Detection', frame)
    cv2.imshow('Lane Segmentation', pred_mask)
    cv2.imshow('Lane Overlay', overlay)

    # ✅ Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release resources
cap.release()
cv2.destroyAllWindows()

