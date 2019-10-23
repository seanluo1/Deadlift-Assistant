import cv2
import time
import numpy as np
import math
#import caffe

def is_adjacent(joint1, joint2):
    joint1_ = joint1.split("_")
    joint2_ = joint2.split("_")
    if joint1_[0] in joint2 or joint1_[1] in joint2:
        return True
    elif joint2_[0] in joint1 or joint2_[1] in joint1:
        return True
    else:
        return False

def angle_calculation(slope1, slope2):
    """
    tau = tan-1(m1-m2/1+m1m2)
    """
    if slope1 == float("inf") and slope2 != 0:
        return math.degrees(math.atan(1/slope2))

    elif slope2 == float("inf") and slope1 !=0 :
        return math.degrees(math.atan(1/slope1))

    elif slope1 == float("inf") and slope2 == 0:
        return 90

    elif slope2 == float("inf") and slope1 == 0:
        return 90

    else:
        angle = math.atan((slope1-slope2)/(1+slope1*slope2))
        # if angle < 0:
        #     angle += 180
        #angle = round(angle, 2)
        return math.degrees(angle)


MODE = "MPI"

if MODE is "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = "./pose/mpi/pose_deploy_linevec.prototxt"
    weightsFile = "./pose/mpi/pose_iter_160000.caffemodel"
    # nPoints = 15
    nPoints = 15
    # POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
    right_deadlift_pose = [[2,3],[3,4], [2,14], [14, 8],  [11,9], [9,10]]
    left_deadlift_pose = [[5, 6], [6, 7], [5, 14], [14, 11], [11, 12], [12, 13], [4,3], [3,5]]



frame = cv2.imread("sean1.jpg")
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

t = time.time()
# input image dimensions for the network
inWidth = 368
inHeight = 368
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

H = output.shape[2]
W = output.shape[3]

# Empty list to store the detected keypoints
points = []

for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > threshold : 
        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else :
        points.append(None)

joint_dictionary = {}
mapping_dictionary = {"5,6":"Elbow_Shoulder",
                      "6,7": "Elbow_Wrist",
                      "5,14": "Shoulder_Chest",
                      "14,11": "Chest_Hip",
                      "11,12": "Hip_Knee",
                      "12,13": "Knee_Ankle",
                      "4,3": "Wrist_Elbow",
                      "3,5": "Elbow_shoulder2"
                      }
joints_angle ={}



# Draw Skeleton
for pair in left_deadlift_pose:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        print(points[partA], points[partB], pair)
        try:
            slope = (points[partB][1] - points[partA][1])/(points[partB][0] - points[partA][0])
        except ZeroDivisionError:
            #slope = 0
            slope = float("inf")
        #print(pair[0],pair[1])
        #print(points[pair[0]], points[pair[1]])
        print(slope)
        joint_dictionary[mapping_dictionary[str(str(pair[0])+ ',' +str(pair[1]))]] = slope
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


print(joint_dictionary)
temp_list = []
for x in joint_dictionary:
    temp_list.append(x)

for x in range(len(temp_list)):
    for y in range(x, len(temp_list)):
        if  temp_list[x] != temp_list[y] and is_adjacent(temp_list[x], temp_list[y]):
            print(temp_list[x], temp_list[y])
            joints_angle[str(temp_list[x])+"_"+str(temp_list[y])] = angle_calculation(joint_dictionary[temp_list[x]], joint_dictionary[temp_list[y]])


print(joints_angle)

cv2.imshow('Output-Keypoints', frameCopy)
cv2.imshow('Output-Skeleton', frame)


cv2.imwrite('Output-Keypoints.jpg', frameCopy)
cv2.imwrite('Output-Skeleton.jpg', frame)

print("Total time taken : {:.3f}".format(time.time() - t))

cv2.waitKey(0)
