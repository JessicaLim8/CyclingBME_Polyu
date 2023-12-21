import cv2

def remove_half_frames(input_file, output_file):
    cap = cv2.VideoCapture(input_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    for i in range(frame_count):
        ret, frame = cap.read()
        if i % 2 == 0:
            writer.write(frame)

    cap.release()
    writer.release()

remove_half_frames("v1_cam1.mp4", "v1_cam1_trim")