import cv2
from ultralytics import YOLO

# Initialize YOLO model
yolo = YOLO("best.pt")  # Update with the actual path

# Open the video file
video_path = r'C:\Users\SALAH MAHMOUD\Downloads\istockphoto-1340525634-640_adpp_is.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
output_video_path = 'output_video_with_objects.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    # Read a frame from the video
    ret, img = cap.read()
    # img = cv2.resize(img, (320, 320))

    if not ret:
        break

    # Perform object detection
    results = yolo.predict(img)

    for box in results[0].boxes.xyxy:
        # Extract individual components
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        # Draw a rectangle around the detected object
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(img)

    # Display the output in a window
    cv2.imshow('Ships', img)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera, writer, and destroy all windows
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved successfully!")
