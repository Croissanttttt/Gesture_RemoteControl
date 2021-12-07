#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np


def convert_relative_to_actual_3d_joint_(param, intrin):
    # Select wrist joint (n) and index finger MCP (m)
    xn, yn = param['keypt'][0] # Wrist
    xm, ym = param['keypt'][9] # Index finger MCP
    xn = (xn-intrin['cx'])/intrin['fx']
    xm = (xm-intrin['cx'])/intrin['fx']
    yn = (yn-intrin['cy'])/intrin['fy']
    ym = (ym-intrin['cy'])/intrin['fy']

    Zn = param['joint'][0,2] # Relative Z coor of wrist
    Zm = param['joint'][9,2] # Relative Z coor of index finger MCP

    # Precalculate value for computing Zroot
    xx = xn-xm
    yy = yn-ym
    xZ = xn*Zn - xm*Zm
    yZ = yn*Zn - ym*Zm
    ZZ = Zn-Zm

    # Compute Zroot relative
    C = 1
    a = xx*xx + yy*yy
    b = 2*(xx*xZ + yy*yZ)
    c = xZ*xZ + yZ*yZ + ZZ*ZZ - C*C
    Zroot = (-b + np.sqrt(b*b - 4*a*c))/(2*a)

    # Convert to actual scale
    s = 0.08 # Note: Hardcode distance from wrist to index finger MCP as 8cm
    Zroot *= s / C

    # Compute actual depth
    param['joint_3d'][:,2] = param['joint'][:,2] + Zroot

    # Compute X and Y
    param['joint_3d'][:,0] = (param['keypt'][:,0]-intrin['cx'])/intrin['fx'] 
    param['joint_3d'][:,1] = (param['keypt'][:,1]-intrin['cy'])/intrin['fy']
    param['joint_3d'][:,:2] *= param['joint_3d'][:,2:3]

    return param['joint_3d']

class GestureRecognition:
    def __init__(self, mode='train'):
        super(GestureRecognition, self).__init__()

        # 11 types of gesture 'name':class label
        self.gesture = {
            'fist':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,
            'rock':7,'spiderman':8,'yeah':9,'ok':10,
        }

        if mode=='train':
            # Create .csv file to log training data
            self.file = open('../data/gesture_train.csv', 'a+')
        elif mode=='eval':
            # Load training data
            file = np.genfromtxt('../data/gesture_train.csv', delimiter=',')
            # Extract input joint angles
            angle = file[:,:-1].astype(np.float32)
            # Extract output class label
            label = file[:, -1].astype(np.float32)
            # Use OpenCV KNN
            self.knn = cv2.ml.KNearest_create()
            self.knn.train(angle, cv2.ml.ROW_SAMPLE, label)


    def train(self, angle, label):
        # Log training data
        data = np.append(angle, label) # Combine into one array
        np.savetxt(self.file, [data], delimiter=',', fmt='%f')
        

    def eval(self, angle):
        # Use KNN for gesture recognition
        data = np.asarray([angle], dtype=np.float32)
        ret, results, neighbours ,dist = self.knn.findNearest(data, 3)
        idx = int(results[0][0]) # Index of class label

        return list(self.gesture)[idx] # Return name of class label

