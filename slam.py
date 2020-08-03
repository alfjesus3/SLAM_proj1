#! /usr/bin/env python
from features_extractor import match_frames, denormalize_pt, Frame
from graph_builder import Map, Point3d
from display import displayVideo
import cv2
import math
import numpy as np
import os

W = 1920 // 2
H = 1080 // 2
F = int(os.getenv("F")) if os.getenv("F") is not None else 270
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

# Global variables
cap = cv2.VideoCapture('./files/drivingCar.mp4')
dp = displayVideo() if os.getenv("Vi") is not None else None
mapp = Map()

def triangulate_projections(m1, m2, pt1, pt2):
    ret = np.zeros((pt1.shape[0], 4))
    pose1 = np.linalg.inv(m1)
    pose2 = np.linalg.inv(m2)

    for i,p in enumerate(zip(pt1,pt2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _,_,vt = np.linalg.svd(A)
        ret[i] = vt[3]

    return ret

def process_frame(frame, mapp):
    if frame.id == 0:
        return 

    f1, f2 = mapp.frames[-1], mapp.frames[-2]
    idx1, idx2, Rt = match_frames(f1, f2) 
    f1.pose = np.dot(Rt, f2.pose)

    for i,idx in enumerate(idx2):
        if f2.pts[idx] is not None:
            f2.pts[idx].add_frame_observation(f1, idx1[i])

    # Converting to 3D point
    pts4d = triangulate_projections(f1.pose, 
                                    f2.pose, 
                                    f1.kpts[idx1], 
                                    f2.kpts[idx2])
    pts4d /= pts4d[:, 3:]

    # rejecting points behind the camera
    unmatched_pts = np.array([f1.pts[i] is None for i in idx1])
    good_pts4 = ((pts4d[:, 2] > 0) & (np.abs(pts4d[:, 3]) > 0.005)) & unmatched_pts
    
    for i, p in enumerate(pts4d):
        if not good_pts4[i]:
            continue
        pt = Point3d(mapp, p)
        pt.add_frame_observation(f1, idx1[i])
        pt.add_frame_observation(f2, idx2[i])

    
    for pt1, pt2 in zip(f1.kpts[idx1], f2.kpts[idx2]):
        u1, v1 = denormalize_pt(pt1, f1.T)
        u2, v2 = denormalize_pt(pt2, f2.T)
        cv2.circle(frame.img, (u1,v1), color = (40, 255, 0), radius = 3)
        cv2.line(frame.img, (u1, v1), (u2, v2), color=(255,0,0))                    

    mapp.display()
    if dp is not None:
        dp.process_frame(frame.img)
    
    if frame.id >= 4:
        mapp.optimize(10)
    

if __name__ == "__main__": 
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            img = cv2.resize(frame, (W,H))
            frame = Frame(mapp, img, K)
            process_frame(frame, mapp)
        else:
            break

