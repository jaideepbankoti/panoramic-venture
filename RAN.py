import numpy as np
import random

"""
    defining the findHomographyMat definition,
    here we also use the ransac method to use the best model
"""
def findHomographyMat(p1, p2):
    A = []
    for i in range(len(p1)):
        x = p1[i][0]
        y = p1[i][1]
        x_ = p2[i][0]
        y_ = p2[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*x_, y*x_, x_])
        A.append([0, 0, 0, -x, -y, -1, x*y_, y*y_, y_])
    A = np.asarray(A)
    U, D, V = np.linalg.svd(A)
    H = V[-1,:].reshape(3,3)/V[-1,-1]
    return H

"""
GeometricDis function calculates the geometric distance between calculated
ones and original ones.
"""
def geometricDis(src_points, dst_points, h):
    p1 = np.transpose(np.array([src_points[0][0], src_points[0][1], 1]))
    p2_calc = np.dot(h, p1)
    p2_calc = (1/p2_calc[2])*p2_calc

    p2 = np.transpose(np.array([dst_points[0][0], dst_points[0][1], 1]))
    err = p2 - p2_calc
    return np.linalg.norm(err)

def ransac(src_points, dst_points, thresh= 0.6):
    maxInliers = []
    H = None
    for i in range(1000):
        """
            find some ranndom points to calculate homography
        """
        src_points1 = src_points[random.randrange(0, len(src_points))]
        dst_points1 = dst_points[random.randrange(0, len(dst_points))]

        src_points2 = src_points[random.randrange(0, len(src_points))]
        dst_points2 = dst_points[random.randrange(0, len(dst_points))]

        randomFourSrc = np.vstack((src_points1, src_points2))
        randomFourDst = np.vstack((dst_points1, dst_points2))

        src_points3 = src_points[random.randrange(0, len(src_points))]
        dst_points3 = dst_points[random.randrange(0, len(dst_points))]
        randomFourSrc = np.vstack((randomFourSrc, src_points3))
        randomFourDst = np.vstack((randomFourDst, dst_points3))

        src_points4 = src_points[random.randrange(0, len(src_points))]
        dst_points4 = dst_points[random.randrange(0, len(dst_points))]
        randomFourSrc = np.vstack((randomFourSrc, src_points4))
        randomFourDst = np.vstack((randomFourDst, dst_points4))

        # calling homography fuction on the random 4 points
        h = findHomographyMat(randomFourSrc, randomFourDst)
        inliers = []

        for i in range(len(src_points)):
            dist = geometricDis(src_points[i],dst_points[i], h)
            if dist< 22:            # set to random tested value to filter by error
                inliers.append((src_points[i],dst_points[i]))

        if len(inliers)> len(maxInliers):
            maxInliers = inliers
            H = h
        if len(maxInliers)> len(src_points)*thresh:
            break
    print('No. of Inliers calculated: ',len(maxInliers))

    # the following if condition checks if the Number of inliers is less than the desired amount, if so it re-evaluates the result
    if(len(maxInliers)<15):         # set to random tested value
        print('no of Inliers was less than 15, Re-evaluating...')
        ransac(src_points, dst_points)
    return H, maxInliers