#!./usr/bin/env python3
from features_extractor import featuresExtractor
from display import displayVideo
import cv2


if __name__ == "__main__":
    cap = cv2.VideoCapture('./files/drivingCar.mp4')
    dp = displayVideo()
    fe = featuresExtractor()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            keypts, _, matches = fe.extracting_feature_matches(frame)
            if matches is not None:
                print(len(matches))

            for p in keypts:
                u, v = map(lambda x: int(round(x)), p.pt)
                cv2.circle(frame, (u,v), color = (40, 255, 0), radius = 3)
            dp.process_frame(frame)
        else:
            break    