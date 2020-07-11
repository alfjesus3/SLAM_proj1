#!./usr/bin/env python3
from features_extractor import featuresExtractor, denormalize_pt
from display import displayVideo
import cv2
import math
import numpy as np

import build.g2o

if __name__ == "__main__":
    cap = cv2.VideoCapture('./files/video.mp4')

    dp = displayVideo()

    W = 1920 // 2
    H = 1080 // 2

    d=1
    cons = math.sqrt(2)/d
    T = np.array([[cons, 0, -W*cons], [0, cons, -H*cons], [0, 0, 1]])

    fe = featuresExtractor(T)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            matches, pose = fe.extracting_feature_matches(frame)
            
            if pose is None: 
                continue

            if matches is not None:
                for pt1, pt2 in matches:
                    u1, v1 = map(lambda x: int(round(x)), pt1)
                    u2, v2 = map(lambda x: int(round(x)), pt2)
                    u1, v1 = denormalize_pt((u1,v1), T)
                    u2, v2 = denormalize_pt((u2,v2), T)
                    cv2.circle(frame, (u1,v1), color = (40, 255, 0), radius = 3)
                    cv2.line(frame, (u1, v1), (u2, v2), color=(255,0,0))                    
            dp.process_frame(frame)
        else:
            break    
