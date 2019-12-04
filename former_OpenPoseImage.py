import cv2
import time
import numpy as np
import math
import pprint
#from nielvis import SPI, DigitalInputOutput, Bank, DIOChannel, SPIClockPhase, SPIClockPolarity, SPIDataDirection

"""
After looking at Professional, 
this is the angles that we have to look at:

{   'Elbow': -7.726617619097228,
    'Hip': 38.516926307102764,
    'Knee': -72.25532837494306,
    'Mid-Back': -17.102728969052375,
    'Shoulder': 46.73570458892839

}

"""

gnd_dead_lift_right_angle_calculations = {
    'Shoulder_Chest to Chest_Hip': 1.0246932979969396,
    'Shoulder_Chest to Shoulder_Elbow': 61.370941432034975,
    'Shoulder_Elbow to Elbow_Wrist': 1.7470702328654104,
    'Chest_Hip to Hip_Knee': -30.50544100980972,
    'Hip_Knee to Knee_Ankle': -85.16844854781782
}


# import caffe

def from_string_back_to_loc(my_string, my_dict):
    # print(my_dict)
    my_result = []
    my_string = my_string.split(" to ")
    # print(my_string)
    first_part = my_string[0]
    second_part = my_string[1]

    for key, value in my_dict.items():
        # print(key,value, my_string[0])
        if value == str(my_string[0]):
            # print("found")
            # print(value)
            pair = key
            pair = pair.split(",")
            my_result.append((int(pair[0]), int(pair[1])))
    for key, value in my_dict.items():
        if value == my_string[1]:
            pair = key
            pair = pair.split(",")
            my_result.append((int(pair[0]), int(pair[1])))

    return my_result


def isback(my_string):
    check_list = ["Chest_Hip","Shoulder_Chest"]
    joints = my_string.split(" to ")
    if joints[0] in check_list and joints[1] in check_list:
        return True
    else:
        return False
    #pass

def from_string_to_loc(my_string, my_dict):
    """
    """
    #print(my_dict)
    my_result = []
    my_string = my_string.split(" to ")
    #print(my_string)
    #first_part = my_string[0]
    #second_part = my_string[1]

    for key, value in my_dict.items():
        #print(key,value, my_string[0])
        if value == str(my_string[0]):
            #print("found")
            #print(value)
            pair = key
            pair = pair.split(",")
            my_result.append( ( int(pair[0]), int(pair[1]) ) )
    for key, value in my_dict.items():
        if value == my_string[1]:
            pair = key
            pair = pair.split(",")
            my_result.append( ( int(pair[0]), int(pair[1]) ) )

    print(my_result)
    return my_result



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
        return math.degrees(math.atan(1 / slope2))

    elif slope2 == float("inf") and slope1 != 0:
        return math.degrees(math.atan(1 / slope1))

    elif slope1 == float("inf") and slope2 == 0:
        return 90

    elif slope2 == float("inf") and slope1 == 0:
        return 90

    else:
        angle = math.atan((slope1 - slope2) / (1 + slope1 * slope2))
        # if angle < 0:
        #     angle += 180
        # angle = round(angle, 2)
        return math.degrees(angle)


MODE = "MPI"

if MODE is "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

elif MODE is "MPI":
    protoFile = "./pose/mpi/pose_deploy_linevec.prototxt"
    weightsFile = "./pose/mpi/pose_iter_160000.caffemodel"
    # nPoints = 15
    nPoints = 15
    # POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                  [11, 12], [12, 13]]
    right_deadlift_pose = [[2, 3], [3, 4], [2, 14], [14, 8], [8, 9], [9, 10]]
    left_deadlift_pose = [[5, 6], [6, 7], [5, 14], [14, 11], [11, 12], [12, 13]]
    #left_deadlift_pose = [[5, 14], [14, 11]]

#frame = cv2.imread("dead_lift_right.png")
# frame = cv2.imread("johan.jpg")
# frame = cv2.imread("model_resize_2.jpg")
# frame = cv2.imread("model_resize_2.jpg")
# frame = cv2.imread("up_right.jpg")
# frame = cv2.imread("gnd_right.jpg")
#frame = cv2.imread("testJohan2.jpg")
frame = cv2.imread("wrong4.jpg")

# frame = cv2.imread("seanblackshirt2.png")
# frame = cv2.imread("shirtless1.png")
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

    if prob > threshold:
        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

joint_dictionary = {}
mapping_dictionary = {"5,6": "Shoulder_Elbow",
                      "6,7": "Elbow_Wrist",
                      "5,14": "Shoulder_Chest",
                      "14,11": "Chest_Hip",
                      "11,12": "Hip_Knee",
                      "12,13": "Knee_Ankle",
                      "2,14":"Neck_Chest"
                      }
mapping_dictionary_right = {
    "2,3": "Shoulder_Elbow",
    "3,4": "Elbow_Wrist",
    "2,14": "Shoulder_Chest",
    "14,8": "Chest_Hip",
    "8,9": "Hip_Knee",
    "9,10": "Knee_Ankle"
}

left_back_dict = {
"14,11":"Chest_Hip",
"5,14":"Shoulder_Chest"
}

joints_angle = {}

# Draw Skeleton
for pair in left_deadlift_pose:
    # for pair in right_deadlift_pose:
    partA = pair[0]
    partB = pair[1]
    # print(pair, partA, partB)

    if points[partA] and points[partB]:
        print(points[partA], points[partB], pair)
        try:
            slope = (points[partB][1] - points[partA][1]) / (points[partB][0] - points[partA][0])
        except ZeroDivisionError:
            # slope = 0
            slope = float("inf")
        # print(pair[0],pair[1])
        # print(points[pair[0]], points[pair[1]])
        #print(slope)
        # <<<<<<< johan_safe_space
        joint_dictionary[mapping_dictionary[str(str(pair[0]) + ',' + str(pair[1]))]] = slope
        #joint_dictionary[mapping_dictionary_right[str(str(pair[0]) + ',' + str(pair[1]))]] = slope
        # =======
        #         joint_dictionary[mapping_dictionary_left[str(str(pair[0]) + ',' + str(pair[1]))]] = slope
        # >>>>>>> master
        #cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        #cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

temp_list = []
#print(joint_dictionary)
for x in joint_dictionary:
    temp_list.append(x)
temp_list.sort()

joint_names_map = {'Chest_Hip to Hip_Knee': 'Hip',
                   'Knee_Ankle to Hip_Knee': 'Knee',
                   'Shoulder_Chest to Chest_Hip': 'Mid-Back',
                   'Elbow_Wrist to Shoulder_Elbow': 'Elbow',
                   'Shoulder_Chest to Shoulder_Elbow': 'Shoulder',
                   'Shoulder_Elbow to Shoulder_Chest': 'Shoulder',
                   'Shoulder_Elbow to Elbow_Wrist': 'Elbow',
                   'Chest_Hip to Shoulder_Chest': 'Mid-Back',
                   'Hip_Knee to Knee_Ankle': 'Knee',
                   'Hip_Knee to Chest_Hip': 'Hip'
                   # =======
                   # joint_names_map = {'Chest_Hip to Hip_Knee': 'Hip1',
                   #                    'Knee_Ankle to Hip_Knee': 'Knee1',
                   #                    'Shoulder_Chest to Chest_Hip': 'Mid-Back1',
                   #                    'Elbow_Wrist to Shoulder_Elbow': 'Elbow1',
                   #                    'Shoulder_Chest to Shoulder_Elbow': 'Shoulder1',
                   #                    'Shoulder_Elbow to Shoulder_Chest': 'Shoulder2',
                   #                    'Shoulder_Elbow to Elbow_Wrist': 'Elbow2',
                   #                    'Chest_Hip to Shoulder_Chest': 'Mid-Back2',
                   #                     'Hip_Knee to Knee_Ankle': 'Knee2',
                   #                     'Hip_Knee to Chest_Hip': 'Hip2'
                   # >>>>>>> master
                   }
#print("here is ")
#print(temp_list)
for x in range(len(temp_list)):
    for y in range(x, len(temp_list)):
        if temp_list[x] != temp_list[y] and is_adjacent(temp_list[x], temp_list[y]):
            #print(temp_list[x], temp_list[y])
            # <<<<<<< johan_safe_space
            joints_angle[str(temp_list[x]) + " to " + str(temp_list[y])] = angle_calculation(
                joint_dictionary[temp_list[x]], joint_dictionary[temp_list[y]])
# =======
#             joints_angle[str(temp_list[x]) + " to " + str(temp_list[y])] = angle_calculation(
#                 joint_dictionary[temp_list[x]], joint_dictionary[temp_list[y]])
#             # joints_angle[joint_names_map[str(temp_list[x]) + " to " + str(temp_list[y])]] = angle_calculation(
#             #     joint_dictionary[temp_list[x]], joint_dictionary[temp_list[y]])
# >>>>>>> master

print("Angle Calculations: ")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(joints_angle)
form = True
for adjacent_joint in joints_angle:
    print(adjacent_joint)
    #print(joints_angle[adjacent_joint])
    if isback(adjacent_joint):
        print("this is back")
        if (abs(joints_angle[adjacent_joint])) < 3: # bad angles
            form = False
            print("bad angles: ", abs(joints_angle[adjacent_joint]))
            #print(adjacent_joint)
            # print(" is close to 0")
            coord = from_string_to_loc(adjacent_joint, left_back_dict)
            partA_1 = coord[0][0]
            partA_2 = coord[0][1]

            partB_1 = coord[1][0]
            partB_2 = coord[1][1]
            print("This is points list: ", points)
            if points[partA_1] and points[partA_2] and points[partB_1] and points[partB_2]:
                # green is 0, 255, 0
                # Red is 0, 0, 255
                #yellow 60, 255, 255
                # cv2.line(frame, points[partA_1], points[partA_2], (0,0, 255), 3, lineType=cv2.LINE_AA)
                cv2.line(frame, points[partA_1], points[partA_2], (0, 0, 255), 4, lineType=cv2.LINE_AA)
                cv2.line(frame, points[partB_1], points[partB_2], (0, 0, 255), 4, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA_1], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partA_2], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB_1], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB_2], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
        else: #good angles
            # print(adjacent_joint)
            coord = from_string_to_loc(adjacent_joint, left_back_dict)
            partA_1 = coord[0][0]
            partA_2 = coord[0][1]

            partB_1 = coord[1][0]
            partB_2 = coord[1][1]
            if points[partA_1] and points[partA_2] and points[partB_1] and points[partB_2]:
                print("Coloring green joints!")
                cv2.line(frame, points[partA_1], points[partA_2], (0, 255, 0), 4, lineType=cv2.LINE_AA)
                cv2.line(frame, points[partB_1], points[partB_2], (0, 255, 0), 4, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA_1], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partA_2], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB_1], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB_2], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
    else: #if not back, color yellow
        coord = from_string_to_loc(adjacent_joint, mapping_dictionary)
        partA_1 = coord[0][0]
        partA_2 = coord[0][1]

        partB_1 = coord[1][0]
        partB_2 = coord[1][1]
        if points[partA_1] and points[partA_2] and points[partB_1] and points[partB_2]:
            # green is 0, 255, 0
            # Red is 0, 0, 255
            # yellow 60, 255, 255
            # cv2.line(frame, points[partA_1], points[partA_2], (0,0, 255), 3, lineType=cv2.LINE_AA)
            cv2.line(frame, points[partA_1], points[partA_2], (60, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(frame, points[partB_1], points[partB_2], (60, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA_1], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partA_2], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB_1], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB_2], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)









cv2.imshow('Output-Keypoints', frameCopy)
cv2.imshow('Output-Skeleton', frame)

cv2.imwrite('Output-Keypoints.jpg', frameCopy)
cv2.imwrite('Output-Skeleton.jpg', frame)

print("Total time taken : {:.3f}".format(time.time() - t))

cv2.waitKey(0)