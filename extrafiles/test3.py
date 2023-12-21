import os
import cv2
import csv
import numpy as np
from obj_dect import ObjectDetection
import math
import datetime

def camera(camNum, od, opath):
    # Open video capture object
    cap = cv2.VideoCapture(camNum)  # 0 represents the default camera connected
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    # Create the video file to name
    i = 0
    while os.path.exists(opath + "/camera%s_%s.avi" % (camNum, i)):
        i += 1
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(opath + "/camera%s_%s.avi" % (camNum, i), fourcc, 4, (int(width), int(height)))

    # Initialize count
    count = 0
    center_points_prev_frame = []
    tracking_objects = {}
    track_id = 0

    # Initialize CSV
    fieldnames = ['time', 'x', 'y']
    with open(opath + "/output_coord.csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.close()

    while True:

        ret, frame = cap.read()
        count += 1
        if not ret:
            print("Error reading frame from Camera 2 video capture")
            break

        # Point current frame
        center_points_cur_frame = []

        # Detect objects on frame
        # (class_ids, scores, boxes) = od.detect(frame)
        (class_ids, scores, boxes) = od.detect(frame)

        for i in range(len(class_ids)):
            if (od.classes[class_ids[i]] != "person"):
                continue
            (x, y, w, h) = boxes[i]
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))
            
            with open(opath + "/output_coord.csv", 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'time': datetime.datetime.now(), 'x': cx, 'y': cy})
                csvfile.close()

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if (len(center_points_cur_frame) and len(center_points_cur_frame[0]) and center_points_cur_frame[0][0] < width / 3):
            break

        # Only at the beginning we compare previous and current frame
        if count <= 2:
            for pt in center_points_cur_frame:
                for pt2 in center_points_prev_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    if distance < 200:
                        tracking_objects[track_id] = pt
                        track_id += 1
        else:

            tracking_objects_copy = tracking_objects.copy()
            center_points_cur_frame_copy = center_points_cur_frame.copy()

            for object_id, pt2 in tracking_objects_copy.items():
                object_exists = False
                for pt in center_points_cur_frame_copy:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    # Update IDs position
                    if distance < 200:
                        tracking_objects[object_id] = pt
                        object_exists = True
                        if pt in center_points_cur_frame:
                            center_points_cur_frame.remove(pt)
                        continue

                # Remove IDs lost
                if not object_exists:
                    tracking_objects.pop(object_id)

            # Add new IDs found
            for pt in center_points_cur_frame:
                tracking_objects[track_id] = pt
                track_id += 1

        for object_id, pt in tracking_objects.items():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            # cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
            text = "(" + str(pt[0]) + ", " + str(pt[1]) + ")"
            cv2.putText(frame, text, (pt[0], pt[1]), 0, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        output.write(frame)

        # Make a copy of the points
        center_points_prev_frame = center_points_cur_frame.copy()

        # If 'q' is pressed, exit the loop
        key = cv2.waitKey(1) & 0xFF

        if key:
            if key == ord('n'):
                break
            if key == ord('x'):
                cv2.destroyAllWindows()
                exit()

    cap.release()
    return

def main():
    i = 0

    opath = "output_videos/session%s" % i

    while os.path.exists(opath):
        i += 1
        opath = "output_videos/session%s" % i

    os.makedirs(opath)

    isCamera1 = True 

    while True:
        if isCamera1:
            camera(0, od, opath)
            #camera("videos/video_trim4.mp4")
            isCamera1 = False
        else:
            camera(1, od, opath)
            #camera("videos/video_trim5.mp4")
            isCamera1 = True

        # If 'q' is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break 

    cv2.destroyAllWindows()

# Initialize Object Detection
od = ObjectDetection()

main()