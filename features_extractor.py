import cv2
import math
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform


def addExtraCoord(pts):
    return np.concatenate([pts, np.ones((pts.shape[0],1))], axis=1)

def normalize_pts(pts, T):
    transPts = addExtraCoord(pts)
    return np.dot(T, np.transpose(transPts)).T[:, 0:2]

def denormalize_pts(pts, T):
    transPts = np.dot(T, addExtraCoord(pts).T)
    transPts = transPts / transPts[2]
    return transPts.T[0:2]

def computeT(pts):
    tmpX = tmpY = 0
    for x,y in pts:
        tmpX += x
        tmpY += y
        
    centroid = (tmpX / len(pts), tmpY / len(pts))

    d=0
    for x,y in pts:
        d += math.sqrt(math.pow(x-centroid[0],2) + math.pow(y-centroid[1],2))
    d /= (len(pts))

    cons = math.sqrt(2)/d
    T  = np.array([[cons, 0, centroid[0]*cons], [0, cons, centroid[1]*cons], [0, 0, 1]])

    return T

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

            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)

            keypts1 = img1.pts[m.queryIdx]
            keypts2 = img2.pts[m.trainIdx]
            res.append((keypts1, keypts2))
    
    res = np.array(res)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    Rt = None
    T1 = computeT(res[:,0])
    T2 = computeT(res[:,1])
    res[:, 0, :] = normalize_pts(res[:, 0, :], T1)
    res[:, 1, :] = normalize_pts(res[:, 1, :], T2)
    
    model, inliers = ransac((res[:, 0], res[:, 1]),
                            EssentialMatrixTransform, 
                            min_samples = 8, 
                            residual_threshold=.005, 
                            max_trials = 200)
    
    res = res[inliers]
    #print(len(res))

    fundM = model.params
    #essenM = np.dot(T2.T, (np.dot(essenM, T1))) #Denormalize Essential Matrix Estimation
    Rt = extract_Rot_trans(fundM)


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
    def __init__(self, mapp, img):
        self.img = img
        self.pose = np.eye(4)

        self.id = len(mapp.frames)
        mapp.frames.append(self)
        
        self.pts, self.des = extract_features_frame(img)


class Point3d(object):
    # It represents a 3d point obtain through the triangulation procedure
    def __init__(self, mapp, loc):
        self.location = loc
        self.frames = []
        self.idxs = []
        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_frame_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)

