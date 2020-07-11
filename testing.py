#!./usr/bin/env python3
from features_extractor import match_frames, denormalize_pt, Frame, Point3d
from display import displayVideo
import cv2
import math
import numpy as np

import build.g2o


def triangulate_projections(m1, m2, pt1, pt2):
    return cv2.triangulatePoints(m1[:3], m2[:3], pt1.T, pt2.T).T

frames = []
def process_frame(frame):
    frames.append(frame)

    if len(frames) < 2:
        return 

    idx1, idx2, Rt = match_frames(frames[-1], frames[-2]) 

    frames[-1].pose = np.dot(Rt, frames[-2].pose)
            
    # Converting to 3D point
    pts4d = triangulate_projections(frames[-1].pose, 
                                    frames[-2].pose, 
                                    frames[-1].pts[idx1], 
                                    frames[-2].pts[idx2])
    pts4d /= pts4d[:, 3:]
            
    # rejecting points behind the camera
    good_pts4 = (pts4d[:, 2] > 0)

    for i, p in enumerate(pts4d):
        if not good_pts4[i]:
            continue
        pt = Point3d(p)
        pt.add_frame_observation(frames[-1], idx1[i])
        pt.add_frame_observation(frames[-2], idx2[i])

    for pt1, pt2 in zip(frames[-1].pts[idx1], frames[-2].pts[idx2]):
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)
        u1, v1 = denormalize_pt((u1,v1), T)
        u2, v2 = denormalize_pt((u2,v2), T)
        cv2.circle(frame.img, (u1,v1), color = (40, 255, 0), radius = 3)
        cv2.line(frame.img, (u1, v1), (u2, v2), color=(255,0,0))                    

    dp.process_frame(frame.img)


if __name__ == "__main__":
    cap = cv2.VideoCapture('./files/video.mp4')

    dp = displayVideo()

    W = 1920 // 2
    H = 1080 // 2

    d=1
    cons = math.sqrt(2)/d
    T = np.array([[cons, 0, -W*cons], [0, cons, -H*cons], [0, 0, 1]])

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            img = cv2.resize(frame, (W,H))
            frame = Frame(img, T)
            process_frame(frame)
        else:
            break    
