import numpy as np

arr = np.array([[-1, 2, 4],[-4, 3, 5],[3, 4, 5]])


def normalize(arr):
    tmp = arr - np.amin(arr)
    tmp = tmp / np.amax(tmp)
    tmp *= 255.0
    return tmp.astype(int)

print(normalize(arr))
# import numpy as np;
 
# # Read image
# im = cv2.imread("/Users/EJ/Desktop/blogsample/1.png", cv2.IMREAD_GRAYSCALE)
 
# # Set up the detector with default parameters.

# params = cv2.SimpleBlobDetector_Params()
# # Change thresholds
# params.minThreshold = 100;    # the graylevel of images
# params.maxThreshold = 255;

# params.filterByColor = True
# params.blobColor = 255

# # Filter by Area
# params.filterByArea = False
# params.minArea = 100000

# detector = cv2.SimpleBlobDetector_create(params)
 
# # Detect blobs.
# keypoints = detector.detect(im)
 
# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# # Show keypoints
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)




# # import os

# # none_RCT = os.listdir('/Volumes/새 볼륨/RCT_DATA/none RCT')
# # print("# of None-RCT patients : ", len(none_RCT))


# # konkuk_RCT_small = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/small sized RCT')
# # RCT_small = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/Small 14개')

# # print("# of konkuk RCT Small : ", len(konkuk_RCT_small))
# # print("# of RCT small : ", len(RCT_small))
# # print("total RCT small : ", len(konkuk_RCT_small) + len(RCT_small))

# # konkuk_RCT_medium = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/medium sized RCT')
# # RCT_medium = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/Medium 30개')

# # print("# of konkuk RCT Small : ", len(konkuk_RCT_medium))
# # print("# of RCT medium : ", len(RCT_medium))
# # print("total RCT medium : ", len(konkuk_RCT_medium) + len(RCT_medium))

# # konkuk_RCT_large = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/large sized RCT')
# # RCT_large = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/Large 15개')

# # print("# of konkuk RCT Large : ", len(konkuk_RCT_large))
# # print("# of RCT large : ", len(RCT_large))
# # print("total RCT large : ", len(konkuk_RCT_large) + len(RCT_large))

# # konkuk_RCT_massive = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/massive RCT')

# # print("total Massive RCT : ", len(konkuk_RCT_massive))


# # print("total : ", len(konkuk_RCT_small) + len(RCT_small) + len(konkuk_RCT_medium) + len(RCT_medium) + len(konkuk_RCT_large) + len(RCT_large) + len(konkuk_RCT_massive))



