import os
import cv2
import csv
import numpy as np
import math
import datetime
from obj_dect import ObjectDetection
import utilities as utls
from DataPoint import DataPoint
from movenet import init_crop_region, run_inference, determine_crop_region, loop_through_people
# from pixel_to_world_mannual import C2PM
# from pixel_to_world_auto import C2PA

opath = "output_videos/session"
campath = "/camera"
filepath = "/output_coord.csv"
fieldnames = ['time', 'x', 'y']
numCameras = 2
xpsAcc = 200
ypsAcc = 50
threshold = 400

def camera(camNum, od, opath):

    if camNum == 0:
        vid = "input_videos/V2_Cam1_lores.MP4"
    if camNum == 1:
        vid = "input_videos/V1_Cam2_lores.MP4"
    # Open video capture object
    cap = cv2.VideoCapture(0)  # 0 represents the default camera connected
    # Open video capture object
    #cap = cv2.VideoCapture(camNum)  # 0 represents the default camera connected
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    image_height, image_width, crop_region = None, None, None

    if opath:
        output_vid, lap = utls.create_vid_path(opath, camNum, width, height)

    # Initialize count
    count = 0
    center_points_prev_frame = []
    tracking_objects = {}
    track_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading Camera Frame %s" % camNum)
            break
        
        (class_ids, scores, boxes) = od.detect(frame)
        center_points_cur_frame = []

        if 0 in class_ids: # check for person
            # Resize image
            img = frame.copy()

            if not image_height and not image_height:
                image_height, image_width, _ = img.shape
                crop_region = init_crop_region(image_height, image_width)
            
            keypoints_with_scores = run_inference(img, crop_region)
            crop_region = determine_crop_region(keypoints_with_scores, image_height, image_width)
            #output_img = draw_prediction_on_image(img, keypoints_with_scores, crop_region=None, close_figure=True, output_image_height=300)
            
            center_points_cur_frame = loop_through_people(frame, keypoints_with_scores, 0.1)

        # If Image Processing
        if opath:
            count += 1

            # Find next point
            newPoint = getMotionTrackPoint(center_points_prev_frame, center_points_cur_frame)
            
            if newPoint:
                center_points_prev_frame.append(newPoint)
                utls.write_to_csv(opath + filepath, fieldnames, newPoint)
            
                # Break if in last quarter of frame
                if (newPoint.getX() < width / 4):
                    print("NEXT SCREEN")
                    return 0

                # cv2.circle(frame, (newPoint.getX(), newPoint.getY()), 5, (0, 0, 255), -1)
                # cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
                text = "(" + str(newPoint.getX()) + ", " + str(newPoint.getY()) + ")"
                # cv2.putText(frame, text, (newPoint.getX(),newPoint.getY()), 0, 1, (0, 0, 255), 2)

        if opath:
            cv2.imshow("Lap %s Camera Frame %s" % (lap, camNum), frame)
            output_vid.write(frame)
        else:
            cv2.imshow("Camera Frame %s - NOT RECORDING" % camNum, frame)
                
        # Check control keys
        key = cv2.waitKey(1) & 0xFF
        if key:
            # Next camera
            if key == ord('n'):
                return 0
            # Start data collection
            if key == ord('s'):
                print("Start")
                return 1
            # End data collection
            if key == ord('e'):
                return 2
            # Exit program
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit()

    cap.release()
    return

def session_initialization():
    # Session Initialization
    outpath = utls.create_session_path(opath)
    utls.create_session_csv(outpath, filepath, fieldnames)
    return outpath

def getMotionTrackPoint(prev_pts, curr_pts):
    if not curr_pts:
        return None
    # HOW TO DETECT FIRST PERSON
    if len(prev_pts) < 1:
        lpoint = max(curr_pts, key=lambda k: k.getX())
        mAcc = max(curr_pts, key=lambda k: k.getAccuracy())
        if lpoint.getAccuracy() >= mAcc.getAccuracy() * 0.8:
            return lpoint
    else:
        # Check if there is a match with the previous point
        newPoint = checkPointMatch(prev_pts, curr_pts, 0)
        if newPoint:
            return newPoint
        
        # Check if there is a match with second previous point
        if len(prev_pts) < 2:
            return None
        if len(prev_pts) < 3:
            newPoint = checkPointMatch(prev_pts, curr_pts, 1)
            if newPoint:
                return newPoint
        else:
            twoFrames = checkPointMatch(prev_pts, curr_pts, 1)
            threeFrames = checkPointMatch(prev_pts, curr_pts, 2)
            if twoFrames == threeFrames:
                return twoFrames
    return None
            
def checkPointMatch(prev_pts, curr_pts, iter):
    trackedPoint = None
    for pt in curr_pts:
        acc = pt.isSamePointAcc(prev_pts[- 1 - iter], xpsAcc, ypsAcc)
        print(f"accuracy is {acc}")
        if acc < threshold:
            if trackedPoint:
                if acc < trackedPoint.isSamePointAcc(prev_pts[- 1 - iter], xpsAcc, ypsAcc):
                    trackedPoint = pt
            else:
                trackedPoint = pt
    return trackedPoint             

def main():
    # Initialize Object Detection
    od = ObjectDetection()

    # Camera Control
    outpath = ''
    camNum = 0
    while True:
        flag = camera(camNum, od, outpath)
        if flag == 0: # next camera
            camNum = (camNum + 1) % numCameras
        elif flag == 1: # start recording
            outpath = session_initialization()
        elif flag == 2: # end recording
            outpath = ''

        # If 'q' is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

main()