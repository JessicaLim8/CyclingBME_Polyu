import cv2
import numpy as np

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, frame = cap.read()

    # Get the frame dimensions
    height, width = frame.shape[:2]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use the Canny edge detection method
    edges = cv2.Canny(gray, 50, 150)

    # Use the Hough Line Transform method
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    print("New Line")
    # Check if any lines were found
    if lines is not None:
        # Draw each line on the frame
        for line in lines:
            x1, y1, x2, y2 = line[0]
            print(line)
            print("coordinates: x1: %s, x2: %s, y1: %s, y2: %s" % (x1, x2, y1, y2))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

def findPoints(img):
    origin = {90, 1000}
    horizontal = {130, 1000}
    vertical = {5, 500}
    return origin, horizontal, vertical
