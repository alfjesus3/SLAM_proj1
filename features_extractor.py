import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform


def addExtraCoord(pts):
    return np.concatenate([pts, np.ones((pts.shape[0],1))], axis=1)

def normalize_pts(pts, T):
    transPt = addExtraCoord(pts)
    return np.dot(np.linalg.inv(T), np.transpose(transPt)).T[:, 0:2]

def denormalize_pt(pt, T):
    cords = np.dot(T, np.array([pt[0], pt[1], 1]))
    cords = cords/ cords[2]
    return int(round(cords[0])), int(round(cords[1]))


class featuresExtractor():
    def __init__(self, t):
        self.prevFr = None
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.T = t


    def extracting_feature_matches(self, img):
        features = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8),
                                        2500, qualityLevel=0.01, minDistance = 3)
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
        
        Rt = None
        print(len(res))
        if len(res) > 0:
            res = np.array(res)
            res[:, 0, :] = normalize_pts(res[:, 0, :], self.T)
            res[:, 1, :] = normalize_pts(res[:, 1, :], self.T)

            model, inliers = ransac((res[:, 0], res[:, 1]),
                                    EssentialMatrixTransform, 
                                    min_samples = 8, 
                                    residual_threshold=.1, 
                                    max_trials = 100)
            res = res[inliers]

            Rt = self.extract_Rot_trans(model.params)

        self.prevFr = {'keypts': keypts, 'descrip': descrip}
        

        return res, Rt
    
    def extract_Rot_trans(self, essen):
        W = np.mat([[0,-1,0], [1,0,0], [0,0,1]], dtype = float)
        U, d, Vt = np.linalg.svd(essen)

        assert np.linalg.det(U) > 0
        if np.linalg.det(Vt) < 0:
            Vt *= -1.0
        R = np.dot(np.dot(U,W), Vt) # rot 1

        if np.sum(R.diagonal()) < 0:
           R = np.dot(np.dot(U,W.T), Vt) # rot 2
        t = U[:, 2]

        Rt = np.eye(4)
        Rt[:3,:3] = R
        Rt[:3, 3] = t
        
        return Rt
        

