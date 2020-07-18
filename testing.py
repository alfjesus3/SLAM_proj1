#! /usr/bin/env python
from features_extractor import match_frames, denormalize_pt, Frame, Point3d
from graph_builder import Map
from display import displayVideo
import cv2
import math
import numpy as np
import os
import g2o


def triangulate_projections(m1, m2, pt1, pt2):
    return cv2.triangulatePoints(m1[:3], m2[:3], pt1.T, pt2.T).T

def process_frame(frame, mapp):
    if len(mapp.frames) < 2:
        return 

    f1, f2 = mapp.frames[-1], mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2) 

    f1.pose = np.dot(Rt, f2.pose)
            
    # Converting to 3D point
    pts4d = triangulate_projections(f1.pose, 
                                    f2.pose, 
                                    f1.pts[idx1], 
                                    f2.pts[idx2])
    pts4d /= pts4d[:, 3:]
    # rejecting points behind the camera
    good_pts4 = (pts4d[:, 2] > 0)
    
    for i, p in enumerate(pts4d):
        if not good_pts4[i]:
            continue
        pt = Point3d(mapp, p)
        pt.add_frame_observation(f1, idx1[i])
        pt.add_frame_observation(f2, idx2[i])

    
    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)
        u1, v1 = denormalize_pt((u1,v1), K)
        u2, v2 = denormalize_pt((u2,v2), K)
        cv2.circle(frame.img, (u1,v1), color = (40, 255, 0), radius = 3)
        cv2.line(frame.img, (u1, v1), (u2, v2), color=(255,0,0))                    

    mapp.display_map()
    if dp is not None:
        dp.process_frame(frame.img)


if __name__ == "__main__":
    cap = cv2.VideoCapture('./files/drivingCar.mp4') # video.mp4

    mapp = Map()

    dp = displayVideo() if os.getenv("DVid") is not None else None

    W = 1920 // 2
    H = 1080 // 2

    """
    d=1
    cons = math.sqrt(2)/d
    T = np.array([[cons, 0, W*cons], [0, cons, H*cons], [0, 0, 1]])
    """
    F = 270
    K = np.array([[F,0, W//2], [0, F, H//2], [0,0,1]])
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            img = cv2.resize(frame, (W,H))
            frame = Frame(mapp, img, K)
            process_frame(frame, mapp)
        else:
            break
