import cv2
import os
import shutil
from ultralytics import YOLO

# 1. Load Model
model = YOLO('yolov8n.pt') 

# 2. Video Path
video_path = r"D:\Instant Advanced AI\Session 14\My Task\Source Video.mp4"
cap = cv2.VideoCapture(video_path)

# --- FOLDER MANAGEMENT ---
folder_path = 'violations'
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
os.makedirs(folder_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30 

entry_frames = {}
car_speeds = {} 
captured_ids = set() 

# Configuration Settings
DISTANCE = 30      
THRESHOLD = 50     
line1_y = 300      
line2_y = 80       
offset = 12        
# --- SNAPSHOT PADDING (To make the crop wider) ---
padding = 40  

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.resize(frame, (640, 360))
    current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)

    results = model.track(frame, persist=True, classes=[2], imgsz=320, verbose=False)

    cv2.line(frame, (0, line1_y), (640, line1_y), (255, 255, 0), 2)
    cv2.line(frame, (0, line2_y), (640, line2_y), (0, 0, 255), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        ids = results[0].boxes.id.int().cpu().tolist()

        for box, id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            y_center = int((y1 + y2) / 2)

            if abs(y_center - line1_y) < offset:
                if id not in entry_frames:
                    entry_frames[id] = current_frame_idx

            if abs(y_center - line2_y) < offset:
                if id in entry_frames:
                    frames_elapsed = current_frame_idx - entry_frames[id]
                    if frames_elapsed > 0:
                        real_time_seconds = frames_elapsed / fps
                        velocity = (DISTANCE / real_time_seconds) * 3.6
                        car_speeds[id] = int(velocity) 

                        if velocity > THRESHOLD and id not in captured_ids:
                            # --- CALCULATING WIDER CROP ---
                            # Ensure coordinates stay within frame boundaries
                            px1 = max(0, x1 - padding)
                            py1 = max(0, y1 - padding)
                            px2 = min(640, x2 + padding)
                            py2 = min(360, y2 + padding)
                            
                            violation_crop = frame[py1:py2, px1:px2]
                            
                            if violation_crop.size != 0:
                                filename = f"{folder_path}/ID_{id}_Speed_{int(velocity)}.jpg"
                                cv2.imwrite(filename, violation_crop)
                                print(f"⚠️ WIDE SNAPSHOT TAKEN: Car {id}")
                                captured_ids.add(id)
                    del entry_frames[id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if id in car_speeds:
                color = (0, 0, 255) if car_speeds[id] > THRESHOLD else (0, 255, 255)
                cv2.putText(frame, f"{car_speeds[id]} km/h", (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("AI Traffic Radar", frame)
    if cv2.waitKey(1) > 0: break

cap.release()
cv2.destroyAllWindows()