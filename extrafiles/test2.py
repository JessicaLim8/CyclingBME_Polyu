import cv2
from matplotlib import pyplot as plt

# Open video capture object
cap = cv2.VideoCapture(2)  # 0 represents the default camera connected

# Check if the video capturing was successful
if not cap.isOpened():
    print("Could not open video capture device")
    exit()

while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Handle the case when frame reading fails
    if not ret:
        print("Error reading frame from video capture")
   
    else:
        # Display the frame
        cv2.imshow("Video", frame)
       
    # If 'q' is pressed, exit the loqop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()