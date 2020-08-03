import cv2
import math
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform


def addExtraCoord(pts):
    return np.concatenate([pts, np.ones((pts.shape[0],1))], axis=1)

def normalize_pts(pts, T):
    transPts = addExtraCoord(pts)
    return np.dot(np.linalg.inv(T), np.transpose(transPts)).T[:, 0:2]

def denormalize_pt(pt, T):
    transPt = np.dot(T, np.array([pt[0], pt[1], 1]).T) 
    transPt = transPt / transPt[2]
    return int(round(transPt[0])), int(round(transPt[1]))

orb = cv2.ORB_create() 
bf = cv2.BFMatcher(cv2.NORM_HAMMING) 

def extract_features_frame(img):
    features = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 
                                        2500,
                                        qualityLevel=0.01, 
                                        minDistance = 3)
    keypts = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]
    keypts, descrip = orb.compute(img, keypts)

    return np.array([(kpt.pt[0], kpt.pt[1]) for kpt in keypts]), descrip

def match_frames(img1, img2):
    res = []
    idx1, idx2 = [], [] 
    # Lowe's ratio test
    matches = bf.knnMatch(img1.des, img2.des, k=2)
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            keypts1 = img1.kpts[m.queryIdx]
            keypts2 = img2.kpts[m.trainIdx]
            
            if np.linalg.norm((keypts1-keypts2)) < 0.1:
                if m.queryIdx not in idx1 and m.trainIdx not in idx2:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)

                    res.append((keypts1, keypts2))

    # Check for no duplicated entries
    assert(len(set(idx1)) == len(idx1))
    assert(len(set(idx2)) == len(idx2))
    
    assert len(res) >= 8
    res = np.array(res)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    Rt = None
    
    model, inliers = ransac((res[:, 0], res[:, 1]),
                            EssentialMatrixTransform, 
                            min_samples = 8, 
                            residual_threshold=.005, 
                            max_trials = 200)
    
    res = res[inliers]
    #print(len(res))

    essenM = model.params
    Rt = extract_Rot_trans(essenM) 

    return idx1[inliers], idx2[inliers], Rt


def extract_Rot_trans(fund):
    W = np.mat([[0,-1,0], [1,0,0], [0,0,1]], dtype = float)
    U, d, Vt = np.linalg.svd(fund)
    
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


class Frame(object):
    def __init__(self, mapp, img, T):
        self.img = img
        self.pose = np.eye(4)
        self.T = T

        self.id = len(mapp.frames)
        mapp.frames.append(self)
        
        kpts, self.des = extract_features_frame(img)
        self.kpts = normalize_pts(kpts, self.T)
        self.pts = [None] * len(self.kpts)

