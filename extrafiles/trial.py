import cv2
# Release the video capture object and close the windows

def camera1():

    while True:
        
        print("Error reading frame from Camera 1 video capture")
        
        key = cv2.waitKey(1) & 0xFF  # Wait for key press

        if key == ord('q'):  # Check if the pressed key is 'q'
            print("hello")
            break  # Exit the loop if 'q' is pressed

    return

def camera2():
    
    while True:
        print("Error reading frame from Camera 2 video capture")

        # If 'q' is pressed, exit the loop
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            print("hello")
            break
    return

isCamera1 = True 

while True:
    if isCamera1:
        camera1()
        isCamera1 = False
    else:
        camera2()
        isCamera1 = True
     
    print("HEY BESTIE")

    # If 'q' is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cv2.destroyAllWindows()
