import os
import csv
import cv2
import datetime

# initialize session path
def create_session_path(opath):
    i = 0
    outpath = opath + str(i)
    while os.path.exists(outpath):
        i += 1
        outpath = opath + str(i)
    os.makedirs(outpath)
    return outpath

# initialize session csv
def create_session_csv(opath, fpath, fieldnames):
    with open(opath + fpath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.close()

def create_vid_path(opath, camNum, width, height):
    # Create the video file to name
    i = 1
    while os.path.exists(opath + "/lap%s_camera%s.avi" % (i, camNum)):
        i += 1
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(opath + "/lap%s_camera%s.avi" % (i, camNum), fourcc, 4, (int(width), int(height)))
    return output, i

def write_to_csv(fpath, fieldnames, dpoint):
    with open(fpath, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'time': dpoint.getTime(), 'x': dpoint.getX(), 'y': dpoint.getY()})
        csvfile.close()