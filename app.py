import cv2
import numpy as np
import json, time

imageItem = {
    "nothing": ["sad.png"],
    "1": ["1b.png","1f.png"],
    "5": ["5b.png","5f.png"],
    "10": ["10b.png","10f.png"],
    "20": ["20b.png","20f.png"],
    "50": ["50b.png","50f.png"],
    "100": ["100b.png","100f.png"],
}

images = {}

for imageType in imageItem:
    for imageLocation in imageItem[imageType]:    
        if imageType in images:
            images[imageType].append(cv2.imread('trainImages/' + imageLocation, 0))
        else:
            images[imageType] = [cv2.imread('trainImages/' + imageLocation, 0)]



# img1 = cv2.imread('trainImages/10f.png', 0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 6)
search_params = dict(checks=50) # or pass empty dictionary
while(True):
    success, img1 = cap.read()
    # originimage = img1.copy()
    # img1 = cv2.cvt

    # orb = cv2.ORB_create()
    sift = cv2.SIFT_create()

    # kp1, des1 = orb.detectAndCompute(img1, None)
    kp1, des1 = sift.detectAndCompute(img1,None)

    bestImage = np.array([])
    bestMatches = []
    for imageType in images:
        for image in images[imageType]:
            # kp2, des2 = orb.detectAndCompute(image, None)
            kp2, des2 = sift.detectAndCompute(image,None)
            
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            
            matches = flann.knnMatch(des1,des2,k=2)

            # matcher = cv2.BFMatcher()
            # matches = matcher.knnMatch(des1, des2, k=2)

            goodMatches = []
            for m,n in matches:
                if m.distance < .45*n.distance:
                    goodMatches.append([m])
            # if len(goodMatches) > 15 and len(goodMatches) > len(bestMatches):
            if len(goodMatches) > len(bestMatches):
                bestImage = image
                bestMatches = goodMatches

    if bestImage.size < 1:
        bestImage = images["nothing"][0]

    matchImage = cv2.drawMatchesKnn(img1, kp1, bestImage, kp2, bestMatches, None, flags=2)


    cv2.imshow("Image", matchImage)
    # cv2.imshow("Image", img1)

    cv2.waitKey(1)