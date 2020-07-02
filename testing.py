#!./usr/bin/env python3
from features_extractor import featuresExtractor
from display import displayVideo
import cv2
import math
import numpy as np


if __name__ == "__main__":
    cap = cv2.VideoCapture('./files/drivingCar.mp4')

    W = 1920 // 2
    H = 1080 // 2

    dp = displayVideo()

    d=1
    cons = math.sqrt(2)/d
    T = np.array([[cons, 0, -W*cons], [0, cons, -H*cons], [0, 0, 1]])
    print(T.shape)

    fe = featuresExtractor(T)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            """
            keypts, _, matches = fe.extracting_feature_matches(frame)
            if matches is not None:
                print(len(matches))

            for p in keypts:
                u, v = map(lambda x: int(round(x)), p.pt)
                cv2.circle(frame, (u,v), color = (40, 255, 0), radius = 3)
            """
            matches = fe.extracting_feature_matches(frame)
            if matches is not None:
                for pt1, pt2 in matches:
                    u1, v1 = map(lambda x: int(round(x)), pt1)
                    u2, v2 = map(lambda x: int(round(x)), pt2)
                    u1, v1 = fe.denormalize_pt((u1,v1))
                    u2, v2 = fe.denormalize_pt((u2,v2))
                    cv2.circle(frame, (u1,v1), color = (40, 255, 0), radius = 3)
                    cv2.line(frame, (u1, v1), (u2, v2), color=(255,0,0))                    
            dp.process_frame(frame)
        else:
            break    