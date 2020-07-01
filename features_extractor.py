import cv2
import matplotlib.pyplot as plt
import numpy as np

class featuresExtractor():
    def __init__(self):
        self.prevFr = None
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher()

    def extracting_feature_matches(self, img):

        features = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 2500, qualityLevel=0.01, minDistance = 3)
        keypts = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]
        keypts, descrip = self.orb.compute(img, keypts)

        matches = None
        if self.prevFr is not None:
            matches = self.bf.match(descrip, self.prevFr['descrip'])

        self.prevFr = {'keypts': keypts, 'descrip': descrip}

        return keypts, descrip, matches
