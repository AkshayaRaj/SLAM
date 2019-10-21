import cv2
import numpy as np
from skimage.transform import FundamentalMatrixTransform
from skimage.measure import ransac

np.set_printoptions(suppress=True)

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

class Extractor(object):
    GX = 16//2
    GY = 12//2

    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.last = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    def normalize(self,pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]
        
    def denormalize(self,  pt): 
        #print(self.Kinv
        ret = np.dot(self.K, np.array([pt[0],pt[1],1]))
        #ret /= ret[2]
        return int(round(ret[0])),int(round(ret[1]))

    def extract(self,img):
        # detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps,des = self.orb.compute(img,kps)

        # matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des,self.last['des'], k=2)
            #print(matches)
            for m,n in matches:
                #print(m,n)
                if m.distance < 0.75 * n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1,kp2))  
                    #print(ret) 
                
        
        # filter
        if len(ret) > 0:
            #print('yes')
            ret = np.array(ret)
            #print(ret.shape)
            #normalize coordinates; subtract to move to 0
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :] )
    

            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    FundamentalMatrixTransform,
                                    min_samples = 8,
                                    residual_threshold = 1,
                                    max_trials = 100
            )
            #print(sum(inliers))
            ret = ret[inliers] 
            s,v,d = np.linalg.svd(model.params) 
            print(v)
            


        #matches = zip([kps[m.queryIdx] for m in matches], [self.last['kps'][m.trainIdx] for m in matches])
        self.last = {'kps': kps, 'des': des}
        return ret
