import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model_path = "best.pt"
model = YOLO(model_path)

# Open the video file
video_path = "videos/fire.mp4"
capture = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
fps = int(capture.get(cv2.CAP_PROP_FPS))

# Define output video writer
output_path = "fire_detected.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    
    # Perform inference
    results = model(frame)
    
    # Draw detection boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            label = result.names[int(box.cls[0].item())]

            if confidence > 0.5:
                color = (0, 255, 0) if confidence > 0.75 else (0, 165, 255) if confidence > 0.6 else (0, 0, 255)
                
                # Draw rectangle with smooth edges
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                
                # Label background
                label_text = f"{label}: {confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 10, y1), color, -1)

                cv2.putText(frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Write the frame to output
    out.write(frame)

    cv2.imshow("Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at {output_path}")