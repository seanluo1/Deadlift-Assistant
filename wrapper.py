import cv2
import serial
import time
import cv2
import time
import numpy as np
import math
import pprint

"""

b'gnd\n'
b'up\n'
b'peak\n'
"""

gnd_right_angle_calculations = {
    'Chest_Hip to Hip_Knee': -77.93022110036928,
    'Chest_Hip to Shoulder_Chest': -8.750610001989925,
    'Elbow_Wrist to Shoulder_Elbow': 14.364582449697206,
    'Hip_Knee to Knee_Ankle': -77.21329855081325,
    'Shoulder_Chest to Shoulder_Elbow': 64.55901135743201
    }


up_right_angle_calculations = {
     'Chest_Hip to Hip_Knee': -89.62869301453506,
    'Chest_Hip to Shoulder_Chest': 2.862405226111749,
    'Elbow_Wrist to Shoulder_Elbow': 0.0,
    'Hip_Knee to Knee_Ankle': 37.31447285161838,
    'Shoulder_Chest to Shoulder_Elbow': -50.19442890773481

}

peak_right_angle_calculations = {
    'Chest_Hip to Hip_Knee': 0.0,
    'Chest_Hip to Shoulder_Chest': 16.213381689985965,
    'Elbow_Wrist to Shoulder_Elbow': -16.829637353189597,
    'Hip_Knee to Knee_Ankle': 14.364582449697206,
    'Shoulder_Chest to Shoulder_Elbow': -2.258872516849116
}



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





def compare(picture, model_dictionary):
    frame = cv2.imread(picture)
    #frame = cv2.imread("shirtless1.png")
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
                          "12,13": "Knee_Ankle"
                          }
    mapping_dictionary_right = {
        "2,3": "Shoulder_Elbow",
        "3,4": "Elbow_Wrist",
        "2,14": "Shoulder_Chest",
        "14,8": "Chest_Hip",
        "8,9": "Hip_Knee",
        "9,10": "Knee_Ankle"
    }

    joints_angle = {}

    # Draw Skeleton
    for pair in right_deadlift_pose:
        # for pair in right_deadlift_pose:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            # print(points[partA], points[partB], pair)

            try:
                slope = (points[partB][1] - points[partA][1]) / (points[partB][0] - points[partA][0])
            except ZeroDivisionError:
                # slope = 0
                slope = float("inf")

            # print(slope)
            joint_dictionary[mapping_dictionary_right[str(str(pair[0]) + ',' + str(pair[1]))]] = [slope, pair]
            #
            # joint_dictionary[mapping_dictionary_right[str(str(pair[0]) + ',' + str(pair[1]))]] = [slope, pair]
            # # cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            # cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    temp_list = []
    for x in joint_dictionary:
        print(x, joint_dictionary[x])
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
                       }
    print("here is the temp list")
    print(temp_list)
    for x in range(len(temp_list)):
        for y in range(x, len(temp_list)):
            if temp_list[x] != temp_list[y] and is_adjacent(temp_list[x], temp_list[y]):
                print(temp_list[x], temp_list[y])
                joints_angle[str(temp_list[x]) + " to " + str(temp_list[y])] = [angle_calculation(
                    joint_dictionary[temp_list[x]][0], joint_dictionary[temp_list[y]][0]),
                    str(temp_list[x]) + " to " + str(temp_list[y]), joint_dictionary[temp_list[x]][1],
                    joint_dictionary[temp_list[y]][1]]
                # joints_angle[joint_names_map[str(temp_list[x]) + " to " + str(temp_list[y])]] = angle_calculation(
                #     joint_dictionary[temp_list[x]], joint_dictionary[temp_list[y]])

    print("Angle Calculations: ")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(joints_angle)
    # pp.pprint(joint_dictionary)

    # Use joint_dictionary to get pair coordinates
    # print(joint_dictionary)
    print(joints_angle)
    form = True
    for adjacent_joint in joints_angle:
        if adjacent_joint in model_dictionary:
            if abs(joints_angle[adjacent_joint][0] - model_dictionary[adjacent_joint]) < 10:
                partA_1 = joints_angle[adjacent_joint][2][0]
                partA_2 = joints_angle[adjacent_joint][2][1]

                partB_1 = joints_angle[adjacent_joint][3][0]
                partB_2 = joints_angle[adjacent_joint][3][1]
                if points[partA_1] and points[partA_2] and points[partB_1] and points[partB_2]:
                    cv2.line(frame, points[partA_1], points[partA_2], (60, 255, 255), 3, lineType=cv2.LINE_AA)
                    cv2.line(frame, points[partB_1], points[partB_2], (60, 255, 255), 3, lineType=cv2.LINE_AA)
                    cv2.circle(frame, points[partA_1], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(frame, points[partA_2], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(frame, points[partB_1], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(frame, points[partB_2], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # bad case, angles not aligned
            else:
                form = False
                # print(adjacent_joint)
                partA_1 = joints_angle[adjacent_joint][2][0]
                partA_2 = joints_angle[adjacent_joint][2][1]

                partB_1 = joints_angle[adjacent_joint][3][0]
                partB_2 = joints_angle[adjacent_joint][3][1]
                if points[partA_1] and points[partA_2] and points[partB_1] and points[partB_2]:
                    cv2.line(frame, points[partA_1], points[partA_2], (0, 0, 255), 3, lineType=cv2.LINE_AA)
                    cv2.line(frame, points[partB_1], points[partB_2], (0, 0, 255), 3, lineType=cv2.LINE_AA)
                    cv2.circle(frame, points[partA_1], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(frame, points[partA_2], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(frame, points[partB_1], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(frame, points[partB_2], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        # if adjacent_joint in mid_way_dictionary:
        #     if abs(joints_angle[adjacent_joint][0] - mid_way_dictionary[adjacent_joint]) < 5:
        #         partA_1 = joints_angle[adjacent_joint][2][0]
        #         partA_2 = joints_angle[adjacent_joint][2][1]
        #
        #         partB_1 = joints_angle[adjacent_joint][3][0]
        #         partB_2 = joints_angle[adjacent_joint][3][1]
        #         if points[partA_1] and points[partA_2] and points[partB_1] and points[partB_2]:
        #             cv2.line(frame, points[partA_1], points[partA_2], (60, 255, 255), 3, lineType=cv2.LINE_AA)
        #             cv2.line(frame, points[partB_1], points[partB_2], (60, 255, 255), 3, lineType=cv2.LINE_AA)
        #             cv2.circle(frame, points[partA_1], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
        #             cv2.circle(frame, points[partA_2], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
        #             cv2.circle(frame, points[partB_1], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
        #             cv2.circle(frame, points[partB_2], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
        #     else:
        #         # print(adjacent_joint)
        #         partA_1 = joints_angle[adjacent_joint][2][0]
        #         partA_2 = joints_angle[adjacent_joint][2][1]
        #
        #         partB_1 = joints_angle[adjacent_joint][3][0]
        #         partB_2 = joints_angle[adjacent_joint][3][1]
        #         if points[partA_1] and points[partA_2] and points[partB_1] and points[partB_2]:
        #             cv2.line(frame, points[partA_1], points[partA_2], (0, 0, 255), 3, lineType=cv2.LINE_AA)
        #             cv2.line(frame, points[partB_1], points[partB_2], (0, 0, 255), 3, lineType=cv2.LINE_AA)
        #             cv2.circle(frame, points[partA_1], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #             cv2.circle(frame, points[partA_2], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #             cv2.circle(frame, points[partB_1], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #             cv2.circle(frame, points[partB_2], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #
        # if adjacent_joint in ending_way_dictionary:
        #     if abs(joints_angle[adjacent_joint][0] - ending_way_dictionary[adjacent_joint]) < 5:
        #         partA_1 = joints_angle[adjacent_joint][2][0]
        #         partA_2 = joints_angle[adjacent_joint][2][1]
        #
        #         partB_1 = joints_angle[adjacent_joint][3][0]
        #         partB_2 = joints_angle[adjacent_joint][3][1]
        #         if points[partA_1] and points[partA_2] and points[partB_1] and points[partB_2]:
        #             cv2.line(frame, points[partA_1], points[partA_2], (60, 255, 255), 3, lineType=cv2.LINE_AA)
        #             cv2.line(frame, points[partB_1], points[partB_2], (60, 255, 255), 3, lineType=cv2.LINE_AA)
        #             cv2.circle(frame, points[partA_1], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
        #             cv2.circle(frame, points[partA_2], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
        #             cv2.circle(frame, points[partB_1], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
        #             cv2.circle(frame, points[partB_2], 8, (60, 255, 255), thickness=-1, lineType=cv2.FILLED)
        #     else:
        #         # print(adjacent_joint)
        #         partA_1 = joints_angle[adjacent_joint][2][0]
        #         partA_2 = joints_angle[adjacent_joint][2][1]
        #
        #         partB_1 = joints_angle[adjacent_joint][3][0]
        #         partB_2 = joints_angle[adjacent_joint][3][1]
        #         if points[partA_1] and points[partA_2] and points[partB_1] and points[partB_2]:
        #             cv2.line(frame, points[partA_1], points[partA_2], (0, 0, 255), 3, lineType=cv2.LINE_AA)
        #             cv2.line(frame, points[partB_1], points[partB_2], (0, 0, 255), 3, lineType=cv2.LINE_AA)
        #             cv2.circle(frame, points[partA_1], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #             cv2.circle(frame, points[partA_2], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #             cv2.circle(frame, points[partB_1], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #             cv2.circle(frame, points[partB_2], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        else:

            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    # cv2.imshow('Output-Keypoints', frameCopy)
    # cv2.imshow('Output-Skeleton', frame)

    # cv2.imwrite('Output-Keypoints.jpg', frameCopy)
    # cv2.imwrite('Output-Skeleton.jpg', frame)

    print("Angle Calculations: ")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(joints_angle)

    #cv2.imshow('Output-Keypoints', frameCopy)

    cv2.imshow("results-keypoints " + picture, frameCopy)
    cv2.imshow("results-skeleton " + picture,frame)

    cv2.imwrite("results-keypoints " + picture, frameCopy)
    cv2.imwrite("results-skeleton " + picture, frame)
    #cv2.imwrite('Output-Keypoints.jpg', frameCopy)


    print("Total time taken : {:.3f}".format(time.time() - t))

    #cv2.waitKey(0)

    return form






port = serial.Serial("/dev/cu.HC-05-DevB", baudrate=9600)
mssge = port.readline().decode("utf-8")
print(mssge)
print("start")

while(mssge != "gnd\n"):
    mssge = port.readline().decode("utf-8")
cam = cv2.VideoCapture(0)
retval, frame = cam.read()
if retval != True:
    raise ValueError("Can't read frame")
cv2.imwrite('final_test_gnd.png', frame)


while(mssge != "up\n"):
    mssge = port.readline().decode("utf-8")
cam = cv2.VideoCapture(0)
retval, frame = cam.read()
if retval != True:
    raise ValueError("Can't read frame")
cv2.imwrite('final_test_up.png', frame)

while(mssge != "peak\n"):
    mssge = port.readline().decode("utf-8")
cam = cv2.VideoCapture(0)
retval, frame = cam.read()
if retval != True:
    raise ValueError("Can't read frame")
cv2.imwrite('final_test_peak.png', frame)

# determine feedback for lift
good_job = True
# compare current photos to ideal model and send code to bt
if not compare('final_test_gnd.png', gnd_right_angle_calculations):
    good_job = False
if not compare('final_test_up.png', up_right_angle_calculations):
    good_job = False
if not compare('final_test_peak.png', peak_right_angle_calculations):
    good_job = False

if good_job:
    port.write(str.encode('1'))
else:
    port.write(str.encode('0'))
# while(True):
#     # port.write(str.encode('1'))
#     # time.sleep(1)
#     # port.write(str.encode('0'))
#     # time.sleep(1)
#     print("while loop started")
#     mssge = port.readline().decode("utf-8")
#     #print(port.readline())
#     print(mssge)
#     if mssge == 'up\n':
#         print("detected up")
#         cam = cv2.VideoCapture(0)
#         retval, frame = cam.read()
#         if retval != True:
#             raise ValueError("Can't read frame")
#
#         cv2.imwrite('final_test_up.png', frame)
#         break
#     if mssge == 'gnd\n':
#         print("detected gnd")
#         cam = cv2.VideoCapture(0)
#         retval, frame = cam.read()
#         if retval != True:
#             raise ValueError("Can't read frame")
#
#         cv2.imwrite('final_test_gnd.png', frame)
#         break
#     if mssge == 'peak\n':
#         print("detected peak")
#         cam = cv2.VideoCapture(0)
#         retval, frame = cam.read()
#         if retval != True:
#             raise ValueError("Can't read frame")
#
#         cv2.imwrite('final_test_peak.png', frame)
#         break
#



# cam = cv2.VideoCapture(0)
# retval, frame = cam.read()
# if retval != True:
#     raise ValueError("Can't read frame")
#
# cv2.imwrite('img2.png', frame)
# cv2.imshow("img1", frame)
# #cv2.waitKey()