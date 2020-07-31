"""
References:
    https://medium.com/@iamhatesz/random-sample-consensus-bd2bb7b1be75
    https://docs.opencv.org/2.4/doc/tutorials/features2d/feature_homography/feature_homography.html
    https://medium.com/analytics-vidhya/image-stitching-with-opencv-and-python-1ebd9e0a6d78
    https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
    https://docs.opencv.org/master/d9/dab/tutorial_homography.html
    https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
    https://github.com/hughesj919/HomographyEstimation/blob/master/Homography.py
    https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
"""

import cv2
import numpy as np
import os                           # this module to access images from folders
import copy                         # this module to create a second copy of the data structure in memory rather than linking
from RAN import ransac              # we develop and import a ransac definition to get inliers
from warp import perspective_warp

"""
This class aids in creating a feat_matcher object just like the BF_matcher object
"""
class feat_matcher:
    def __init__(self):
        self._match = []  # a list to store all matches

    def euclidean_dist(self, a, b):  # a method to calculate euclidean distance given a numpy array
        return np.sqrt(np.sum((a - b) ** 2))

    def matchQueryDes(self, des1, des2):
        self._match.clear()
        qidx, tidx = 0,0
        for i in range(len(des1)):
            dist = float('inf')                                     # this sets the value of distance to infinity at each entry to loop
            for j in range(len(des2)):
                temp = self.euclidean_dist(des1[i], des2[j])
                if temp < dist:                 # here the match is included only if it satisfies the following condition
                    dist = temp
                    qidx, tidx = i, j
            self._match.append(cv2.DMatch(qidx, tidx, dist))
        return self._match

"""
The match_query fuction takes in query and train descriptors as the argument and return sorted matches by calling another function
fromt the feat_matcher class
"""
def match_query(desQ, desDB):
    # create a BFmatcher object
    fm = feat_matcher()

    # Match descriptors
    matches = fm.matchQueryDes(desQ, desDB)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    return matches

"""
    defining the findHomographyMat definition,
    here we also use the ransac method to use the best model in the main stitching code below!
"""
def findHomographyMat(p1, p2):
    """
    :param p1: point in the first image
    :param p2: point from the second image
    :return: homography matrix H
    """
    A = []
    for i in range(len(p1)):
        x = p1[i][0][0]
        y = p1[i][0][1]
        x_ = p2[i][0][0]
        y_ = p2[i][0][1]
        A.append([-x, -y, -1, 0, 0, 0, x*x_, y*x_, x_])
        A.append([0, 0, 0, -x, -y, -1, x*y_, y*y_, y_])
    A = np.asarray(A)
    U, D, V = np.linalg.svd(A)
    H = V[-1,:].reshape(3,3)/V[-1,-1]
    return H


def main(DIR):
    img_list = os.listdir(DIR)
    sift = cv2.xfeatures2d.SIFT_create()

    # let us now split the list into two, leftList and rightList
    leftList, rightList = [], []
    for i in range(len(img_list)):
        if i+1 <= len(img_list)//2:
            leftList.append(img_list[i])
        else:
            rightList.append(img_list[i])

    """
    We shall start with stitching right to left, picking images from rightList
    """

    temp_list = copy.deepcopy(rightList)                    # here copy module used to make a secondary copy of the list
    imgR = cv2.imread('{}/{}'.format(DIR, temp_list[-1]))   # to read from the directory and from end of temp_list
    del temp_list[-1]

    while(len(temp_list) != 0 ):
        """
            The leftward stitching loop
        """
        print('Leftward Stitching')
        imgL = cv2.imread('{}/{}'.format(DIR, temp_list[-1]))
        del temp_list[-1]
        imgR_g = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgL_g = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        # finding the matching keypoint in both images
        kp1, des1 = sift.detectAndCompute(imgR_g, None)
        kp2, des2 = sift.detectAndCompute(imgL_g, None)

        matches = match_query(des1, des2)
        good_matches = matches[:50]

        # here the corresponding matches are mapped, uncomment the following few lines to view them
        """img3 = cv2.drawMatches(imgR, kp1, imgL, kp2, good_matches, None, flags=2)
        cv2.imshow('mathced image', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        h, maxInliers = ransac(src_pts, dst_pts)
        p1 = np.array([val[0] for val in maxInliers])
        p2 = np.array([val[1] for val in maxInliers])
        H = findHomographyMat(p1, p2)
        # H = findHomographyMat(src_pts, dst_pts)                               # used as reference in finding self implemented homography without ransac
        # H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)          # used as reference in finding homography via inbuilt method

        """
            The following few lines uses the warp perspective method and projects the image on a different canvas with new dimension
        """
        width, height = imgR.shape[1]+ imgL.shape[1], imgR.shape[0]
        dst = perspective_warp(imgR, H, (width, height))
        """cv2.imshow('asdf', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
        dst[0:imgL.shape[0], 0:imgL.shape[1]] = imgL
        imgR = dst

    print('------------------------------------------------------------------------------------------------------')

    FinalRightImg = imgR        # this is the final right image that will be stitched to the left half
    cv2.imshow("Leftward Stitched Slice", FinalRightImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    Now lets perform the same image stitching procedure left to right on the leftList
    """
    temp_list = copy.deepcopy(leftList)
    imgL = cv2.imread('{}/{}'.format(DIR, temp_list[0]))
    del temp_list[0]
    while(len(temp_list) != 0 ):
        """
                    The rightward stitching loop
        """
        print('Rightward Stitching')
        imgR = cv2.imread('{}/{}'.format(DIR, temp_list[0]))
        del temp_list[0]
        imgL = cv2.flip(imgL, flipCode=1)
        imgR = cv2.flip(imgR, flipCode=1)
        imgR_g = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgL_g = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        # finding the matching keypoint in both images
        kp1, des1 = sift.detectAndCompute(imgL_g, None)
        kp2, des2 = sift.detectAndCompute(imgR_g, None)

        matches = match_query(des1, des2)
        good_matches = matches[:40]

        # here the corresponding matches are mapped, uncomment the following few lines
        """img3 = cv2.drawMatches(imgL, kp1, imgR, kp2, good_matches, None, flags=2)
        cv2.imshow('mathced image', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        h, maxInliers = ransac(src_pts, dst_pts)
        p1 = np.array([val[0] for val in maxInliers])
        p2 = np.array([val[1] for val in maxInliers])
        H = findHomographyMat(p1, p2)
        # H = findHomographyMat(src_pts, dst_pts)                               # used as reference in finding self implemented homography without ransac
        # H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)          # used as reference in finding homography via inbuilt method

        """
                    The following few lines uses the warp perspective method and projects the image on a different canvas with new dimension
                    also we flip the image as we are doing the merge similar to leftward stitch but flip the result to make it look like we
                    made a right stitch
        """
        width, height = imgR.shape[1]+ imgL.shape[1], imgL.shape[0]
        dst = perspective_warp(imgL, H, (width, height))
        dst[0:imgR.shape[0], 0:imgR.shape[1]] = imgR
        dst = cv2.flip(dst, flipCode=1)
        imgL = dst

    print('------------------------------------------------------------------------------------------------------')

    FinalLeftImg = imgL
    cv2.imshow("Rightward stitched image", FinalLeftImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    Final stitch code, we will be left stitching the FinalRightImg to the FinalLeftImg 
    """
    print('Final Stitching')
    FinalRightImg_g = cv2.cvtColor(FinalRightImg, cv2.COLOR_BGR2GRAY)
    FinalLeftImg_g = cv2.cvtColor(FinalLeftImg, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(FinalRightImg_g, None)
    kp2, des2 = sift.detectAndCompute(FinalLeftImg_g, None)

    matches = match_query(des1, des2)
    good_matches = matches[:40]

    # here the corresponding matches are mapped, uncomment the following few lines
    """img3 = cv2.drawMatches(FinalRightImg, kp1, FinalLeftImg, kp2, good_matches, None, flags=2)
    cv2.imshow('mathced image', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    h, maxInliers = ransac(src_pts, dst_pts)
    p1 = np.array([val[0] for val in maxInliers])
    p2 = np.array([val[1] for val in maxInliers])
    H = findHomographyMat(p1, p2)
    # H = findHomographyMat(src_pts, dst_pts)                               # used as reference in finding self implemented homography without ransac
    # H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)          # used as reference in finding homography via inbuilt method

    # ----------the warp perspective method-----------------------
    width, height = FinalRightImg.shape[1] + FinalLeftImg.shape[1], FinalRightImg_g.shape[0]
    dst = perspective_warp(FinalRightImg, H, (width, height))
    dst[0:FinalLeftImg.shape[0], 0:FinalLeftImg.shape[1]] = FinalLeftImg

    cv2.imshow("Final Stitched Image", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # after the whole algorithm is finished running the final image is saved up in the working directory!
    cv2.imwrite((DIR+".png"), dst)

if __name__ == "__main__":
    """
        In the next code segment all the images from a directory is read and stored as an image variable,
        You can replace the string fed to DIR by the following choices:
            'CollegeMain'
            'Construction'
            'NearMinar'
            'Room'
            'SolarPath'
    """
    DIR = "Construction"
    main(DIR)