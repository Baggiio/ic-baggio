import pyopenpose as op
import cv2
import time
import sys
import os
import json
import math

# keypoints_list[person][keypoint][x, y, score]
# check keypoints indexes in refs directory

def convertDegreeToRadian(degree):
    return ((degree*math.pi)/180)

def convertRadianToDegree(radian):
    return ((radian*180)/math.pi)

def anglesForKnees(hip, knee, ankle):

    if (hip[0]==0 and hip[1]==0) or (knee[0]==0 and knee[1]==0) or (ankle[0]==0 and ankle[1]==0):
        return 0;
    
    ax=hip[0] - knee[0]
    ay=hip[1] - knee[1]
    a = math.sqrt(ax**2 + ay**2)

    bx=knee[0] - ankle[0]
    by=knee[1] - ankle[1]
    b = math.sqrt(bx**2 + by**2)
    
    cx=hip[0] - ankle[0]
    cy=hip[1] - ankle[1]
    c = math.sqrt(cx**2 + cy**2)

    cos0 = c**2 - a**2 - b**2
    cos0 = cos0/(-2*a*b)

    inverse_cos0 = math.acos(cos0)
    
    return convertRadianToDegree(inverse_cos0)

try:
    params = dict()
    params["model_folder"] = "/home/lab-03/Downloads/openpose/models/"

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    video_list = os.listdir('media')
    video_list = [video for video in video_list if video.endswith('.mp4')]
    video_list.sort()
    print("Video list: " + str(video_list))

    for video in video_list:
        cap = cv2.VideoCapture('media/' + video)

        start = time.time()

        print("Processing video: " + video)

        if not os.path.exists("output_frames"):
            os.makedirs("output_frames")

        with open("output_frames/" + video[:-4] + "_output.txt", "w") as outfile:
            outfile.write("")

        move_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                datum = op.Datum()
                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                left_cam = []
                right_cam = []
                front_cam = []

                for i in range(len(datum.poseKeypoints)):
                    if datum.poseKeypoints[i][1][0] < 640:
                        left_cam.append(datum.poseKeypoints[i])

                    elif datum.poseKeypoints[i][1][0] > 1280:
                        right_cam.append(datum.poseKeypoints[i])
                    
                    elif datum.poseKeypoints[i][1][0] > 640 and datum.poseKeypoints[i][1][0] < 1280:
                        front_cam.append(datum.poseKeypoints[i])

                # remove myself from right cam (only in this particular case)
                if len(right_cam) > 0:
                    if right_cam[0][0][0] > right_cam[1][0][0]:
                        right_cam.pop(1)
                    else:
                        right_cam.pop(0)

                if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:
                    with open("output_frames/" + video[:-4] + "_points.txt", "w") as outfile:
                        outfile.write("Number of people: " + str(len(datum.poseKeypoints)) + "\n")
                        outfile.write("Person 1 (left)\n")
                        outfile.write(str(left_cam[0][0]) + "\n\n")
                        outfile.write("Person 2 (center)\n")
                        outfile.write(str(front_cam[0][0]) + "\n\n")
                        outfile.write("Person 3 (right)\n")
                        outfile.write(str(right_cam[0][0]) + "\n\n")

                progress = (cap.get(cv2.CAP_PROP_POS_FRAMES) + 1) / cap.get(cv2.CAP_PROP_FRAME_COUNT)
                sys.stdout.write('\r')
                # the exact output you're looking for:
                sys.stdout.write("Processing frames: [%-20s] %d%%" % ('='*int(20*progress), 100*progress))
                sys.stdout.flush()

                # print("Body keypoints: \n" + str(datum.poseKeypoints))
                try:
                    knee_angle_right_side = anglesForKnees(left_cam[0][9], left_cam[0][10], left_cam[0][11])
                    knee_angle_left_side = anglesForKnees(right_cam[0][12], right_cam[0][13], right_cam[0][14])
                    right_knee_angle_front = anglesForKnees(front_cam[0][9], front_cam[0][10], front_cam[0][11])
                    left_knee_angle_front = anglesForKnees(front_cam[0][12], front_cam[0][13], front_cam[0][14])

                    if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:

                        saved_side_angles = [knee_angle_right_side, knee_angle_left_side]

                        lowest_left_knee_angle = left_knee_angle_front
                        lowest_right_knee_angle = right_knee_angle_front
                        highest_left_knee_angle = left_knee_angle_front
                        highest_right_knee_angle = right_knee_angle_front
                        left_knee_lowest_and_highest = []
                        right_knee_lowest_and_highest = []

                    elif cap.get(cv2.CAP_PROP_POS_FRAMES) > 3:
                    # print("Frame: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    # print("Knee angle right side (left cam): " + str(knee_angle_right_side))
                    # print("Knee angle left side (right cam): " + str(knee_angle_left_side))
                    # print("Right knee angle (front cam): " + str(right_knee_angle_front))
                    # print("Left knee angle (front cam): " + str(left_knee_angle_front))
                    # print("")

                        if (knee_angle_right_side < 160 or knee_angle_left_side < 160) and saved_side_angles[0] >= 160 and saved_side_angles[1] >= 160:
                            move_counter += 1

                        if (knee_angle_right_side > 160 or knee_angle_left_side > 160) and saved_side_angles[0] <= 160 and saved_side_angles[1] <= 160:
                            left_knee_lowest_and_highest.append([lowest_left_knee_angle, highest_left_knee_angle])
                            right_knee_lowest_and_highest.append([lowest_right_knee_angle, highest_right_knee_angle])
                            lowest_left_knee_angle = left_knee_angle_front
                            lowest_right_knee_angle = right_knee_angle_front
                            highest_left_knee_angle = left_knee_angle_front
                            highest_right_knee_angle = right_knee_angle_front


                        if (right_knee_angle_front < 160 or left_knee_angle_front < 160):

                            if right_knee_angle_front < lowest_right_knee_angle:
                                lowest_right_knee_angle = right_knee_angle_front
                            
                            if left_knee_angle_front < lowest_left_knee_angle:
                                lowest_left_knee_angle = left_knee_angle_front

                            if right_knee_angle_front > highest_right_knee_angle:
                                highest_right_knee_angle = right_knee_angle_front

                            if left_knee_angle_front > highest_left_knee_angle:
                                highest_left_knee_angle = left_knee_angle_front

                            with open("output_frames/" + video[:-4] + "_output.txt", "a") as outfile:
                                outfile.write("Move counter: " + str(move_counter) + "\n")
                                outfile.write("Frame: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)) + "\n")
                                outfile.write("Knee angle right side (left cam): " + str(knee_angle_right_side) + "\n")
                                outfile.write("Knee angle left side (right cam): " + str(knee_angle_left_side) + "\n")
                                outfile.write("Right knee angle (front cam): " + str(right_knee_angle_front) + "\n")
                                outfile.write("Left knee angle (front cam): " + str(left_knee_angle_front) + "\n")
                                outfile.write("\n")

                        saved_side_angles = [knee_angle_right_side, knee_angle_left_side]
                except:
                    print("Error in calculating knee angles")
                    pass

                if not os.path.exists("output_frames/" + video[:-4]):
                    os.makedirs("output_frames/" + video[:-4])

                cv2.imwrite("output_frames/" + video[:-4] + "/output_" + str(cap.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", datum.cvOutputData)
                
                cv2.imshow("OpenPose 1.7.0", datum.cvOutputData)
                key = cv2.waitKey(15)
                if key == 27: break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        end = time.time()
        print("\nOpenPose successfully finished. Total time: %.2f seconds" % (end - start))
        print("\nMove counter: " + str(move_counter))
        
        for i in range(len(left_knee_lowest_and_highest)):
            print("\nMove %d: Left knee lowest: %.2f° | Left knee highest: %.2f° | Right knee lowest: %.2f° | Right knee highest: %.2f°" % (i + 1, left_knee_lowest_and_highest[i][0], left_knee_lowest_and_highest[i][1], right_knee_lowest_and_highest[i][0], right_knee_lowest_and_highest[i][1]))
            print("Right knee difference: %.2f°" % (right_knee_lowest_and_highest[i][1] - right_knee_lowest_and_highest[i][0]))
            print("Left knee difference: %.2f°" % (left_knee_lowest_and_highest[i][1] - left_knee_lowest_and_highest[i][0]))
        
except Exception as e:
    print(e)
    sys.exit(-1)