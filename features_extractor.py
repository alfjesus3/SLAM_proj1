import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

class featuresExtractor():
    def __init__(self, t):
        self.prevFr = None
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.T = t

    def addExtraCoord(self, pts):
        return np.concatenate([pts, np.ones((pts.shape[0],1))], axis=1)

    def normalize_pts(self, pts):
        transPt = self.addExtraCoord(pts)
        return np.dot(np.linalg.inv(self.T), np.transpose(transPt)).T[:, 0:2]

    def denormalize_pt(self, pt):
        cords = np.dot(self.T, np.array([pt[0], pt[1], 1]))
        return int(round(cords[0])), int(round(cords[1]))


    def extracting_feature_matches(self, img):

        features = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 2500, qualityLevel=0.01, minDistance = 3)
        keypts = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]
        keypts, descrip = self.orb.compute(img, keypts)
        res = []
        
        if self.prevFr is not None:
            matches = self.bf.knnMatch(descrip, self.prevFr['descrip'], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    keypts1 = keypts[m.queryIdx].pt
                    keypts2 = self.prevFr['keypts'][m.trainIdx].pt
                    res.append((keypts1, keypts2))

        if len(res) > 0:
            res = np.array(res)
            res[:, 0, :] = self.normalize_pts(res[:, 0, :])
            res[:, 1, :] = self.normalize_pts(res[:, 1, :])

            model, inliers = ransac((res[:, 0], res[:, 1]), FundamentalMatrixTransform, min_samples = 8, residual_threshold=.1, max_trials = 100)
            res = res[inliers]

        self.prevFr = {'keypts': keypts, 'descrip': descrip}

        return res #keypts, descrip, matches
