#!/usr/bin/env python

'''
Feature-based image matching sample.

USAGE
  find_obj.py [--feature=<sift|surf|orb>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf of orb. Append '-flann' to feature name
                to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

import numpy as np
import os
import cv2
from common import anorm, getsize
from video import create_capture
import time
import Tkinter
import tkFileDialog

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6


def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.SIFT()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.SURF(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB(400)
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def explore_match(win, img_arr,imageNames, img2, kp_pairs_arr, status_arr = None, H_arr = None):
    # tester
    # img1 = img_arr[1]
    # kp_pairs = kp_pairs_arr[1]
    # status = status_arr[1]
    # H = H_arr[1]

    # store each images' height & width
    h_arr = []
    w_arr = []
    
    for i in range(0,len(img_arr)):
        h1, w1 = img_arr[i].shape[:2]
        h_arr.append(h1)
        w_arr.append(w1)
    
    h2, w2  = img2.shape[:2]
    h_arr_sum = np.sum(h_arr)
    vis = np.zeros((max(h_arr_sum, h2), max(w_arr)+w2 ), np.uint8)

    #store each images' y location except the 1st
    loc_y_arr =[]

    for i in range(0,len(img_arr)):
        loc_y_arr.append(sum(h_arr[:i]))

    #locate image
    for i in range(0,len(img_arr)):
        vis[loc_y_arr[i]:h_arr[i]+loc_y_arr[i], :w_arr[i]] = img_arr[i]


    vis[:h2,  max(w_arr): max(w_arr)+w2 ] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for i in range(0,len(img_arr)):
        if H_arr[i] is not None:
            corners = np.float32([[0, 0], [w_arr[i], 0], [w_arr[i], h_arr[i]], [0,  h_arr[i]]])
            corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H_arr[i]).reshape(-1, 2) + (max(w_arr), 0) )
            cv2.polylines(vis, [corners], True, (255, 255, 255))
            cv2.putText(vis,imageNames[i],tuple(corners[0]),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)

        # if status_arr[i] is None:
        #     status_arr[i] = np.ones(len(kp_pairs_arr[i]), np.bool_)

        # try:
        #     p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
        #     p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w_arr[i], 0)

        #     green = (0, 255, 0,120)
        #     red = (0, 0, 255)
        #     white = (255, 255, 255)
        #     kp_color = (51, 103, 236)
        #     for (x1, y1), (x2, y2), inlier in zip(p1, p2, status_arr[i]):
        #         if inlier:
        #             col = green
        #             cv2.circle(vis, (x1, y1), 2, col, -1)
        #             cv2.circle(vis, (x2, y2), 2, col, -1)
        #         else:
        #             col = red
        #             r = 2
        #             thickness = 3
        #             cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
        #             cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
        #             cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
        #             cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
        #     vis0 = vis.copy()
        #     for (x1, y1), (x2, y2), inlier in zip(p1, p2, status_arr[i]):
        #         if inlier:
        #             cv2.line(vis, (x1, y1), (x2, y2), green)
        # except:
        #     pass
    cv2.imshow(win, vis)
    return vis

def trainImage(filenames, detector):
    image_return = []
    kp_return = []
    des_return = []

    for fn in filenames:
        image_t = cv2.imread(fn,0)
        kp_t, desc_t = detector.detectAndCompute(image_t, None)
        image_return.append(image_t)
        kp_return.append(kp_t)
        des_return.append(desc_t)
    
    return image_return,kp_return, des_return


def openFile():
 
    Tkinter.Tk().withdraw() # Close the root window
    in_path = tkFileDialog.askopenfilename()
    print in_path

if __name__ == '__main__':
    print __doc__

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    
    try:
        img_dir = args[0]
    except:
        img_dir = 'images'

    imageNames = os.listdir(img_dir)
    imageFiles = []

    for i in range(0, len(imageNames)):
        imageFiles.append(img_dir+'/'+imageNames[i])

    detector, matcher = init_feature(feature_name)

    if detector != None:
        print 'using', feature_name
    else:
        print 'unknown feature:', feature_name
        sys.exit(1)

    img_arr, kp_arr, desc_arr = trainImage(imageFiles,detector)
    
    
    try: video_src = video_src[0]
    except: video_src = 0
    cam = create_capture(video_src)

   
    # video feed
    while True:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (int(360*1.5), int(240*1.5))) # resize to find proper speed & size of cam window
        
        kp2, desc2 = detector.detectAndCompute(frame, None)
        kp_pairs_arr = []
        status_arr = []
        H_arr = []
        
        for i in range(0,len(desc_arr)):
            raw_matches = matcher.knnMatch(desc_arr[i], trainDescriptors = desc2, k = 2) #2
            p1, p2, kp_pairs = filter_matches(kp_arr[i], kp2, raw_matches)
            
            kp_pairs_arr.append(kp_pairs)
            if len(p1) >= 12:
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                print '%d / %d  inliers/matched' % (np.sum(status), len(status))
            else:
                H, status = None, None
                print '%d matches found, not enough for homography estimation' % len(p1)
            status_arr.append(status)
            H_arr.append(H)

        vis = explore_match('find_obj', img_arr,imageNames, frame, kp_pairs_arr, status_arr, H_arr)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
    

    # static image comparison
    def match_and_draw(win):
        print 'matching...'
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            #print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        else:
            H, status = None, None
            #print '%d matches found, not enough for homography estimation' % len(p1)

        vis = explore_match(win, img1, img2, kp_pairs, status, H)
        

    #match_and_draw('find_obj')
    cv2.destroyAllWindows()
