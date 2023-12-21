import cv2
# Release the video capture object and close the windows

def camera1():
    # Open video capture object
    cap1 = cv2.VideoCapture(0)  # 0 represents the default camera connected

    # Check if the video capturing was successful
    # if not cap1.isOpened():
    #     print("Could not open Camera 1")
    #     exit()

    while True:
        # Read frame from video capture
        ret, frame = cap1.read()
        
        # Handle the case when frame reading fails
        if not ret:
            print("Error reading frame from Camera 1 video capture")
            break

        else:
            # Display the frame
            cv2.imshow("Video 1", frame)
        
        # If 'q' is pressed, exit the loop
        key = cv2.waitKey(1) & 0xFF

        if key:
            if key == ord('n'):
                break
            if key == ord('x'):
                cv2.destroyAllWindows()
                exit()

    cap1.release()
    return

def camera2():
    # Open video capture object
    cap2 = cv2.VideoCapture(1)  # 0 represents the default camera connected

    # # Check if the video capturing was successful
    # if not cap2.isOpened():
    #     print("Could not open Camera 2")
    #     exit()

    while True:
        # Read frame from video capture
        ret, frame = cap2.read()
        
        # Handle the case when frame reading fails
        if not ret:
            print("Error reading frame from Camera 2 video capture")
            break

        else:
            # Display the frame
            cv2.imshow("Video 2", frame)
        
        # If 'q' is pressed, exit the loop
        key = cv2.waitKey(1) & 0xFF

        if key:
            if key == ord('n'):
                break
            if key == ord('x'):
                cv2.destroyAllWindows()
                exit()

    cap2.release()
    return

isCamera1 = True 

while True:
    if isCamera1:
        camera1()
        isCamera1 = False
    else:
        camera2()
        isCamera1 = True

    # If 'q' is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break 

cv2.destroyAllWindows()
