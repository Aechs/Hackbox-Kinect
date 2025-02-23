#!/usr/bin/env python
# -*- coding: utf-8 -*-
# as if 2/23/2025 9:30am after submission, I had found a project that implements image detection in the web and seems to update the hand tracking points faster -> https://github.com/handtracking-io/yoha/tree/main?tab=read
import csv
import copy
import argparse
import itertools
from time import time, sleep
from math import fabs, sqrt, pow
from collections import Counter
from collections import deque
from pynput.mouse import Button # type: ignore
from pynput.keyboard import Key
import pynput
from pygame import mixer
import cv2 as cv
import numpy as np
import mediapipe as mp
import socket

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

control_state = 0  # used as a state value to signify that the mouse control is active

# Initialize mouse controller
mouse = pynput.mouse.Controller()
keyboard = pynput.keyboard.Controller()

# Constants
click_to_on_threshold = 30.00
click_to_off_threshold = 80.00
middle_to_ring_threshold = 0.20
middle_to_ring_time = 0.75 # seconds
palm_gesture_center_move_x_threshold = 10
palm_gesture_center_move_y_threshold = 10

# Helper variables
last_sign = -1
mouse_pressed = False
hand_found = False
index_to_thumb = 0.00
hand_base_gesture = 0  # this score is used to count up every frame that the hand is in the base position
hand_base_gesture_started = False  # this is set to true when the timer starts, false otherwise
current_threshold = click_to_on_threshold
base_gesture_active = False  # True when the user has their palm up to the camera for 2ish seconds
middle_finger_slope = 0
ring_finger_slope = 0
palm_position = (0, 0) # the center of the palm
last_palm_position = [0, 0]
palm_gesture_radius_counter = 0

# Initialize sounds
mixer.init()
alert = mixer.Sound("C:\\Windows\\Media\\notify.wav")
gesture_started = mixer.Sound("C:\\Windows\\Media\\Windows Proximity Notification.wav")

def data_update(landmarks: list, hand_sign: int):
    global mouse_pressed
    global last_sign
    global index_to_thumb
    global hand_base_gesture
    global current_threshold
    global hand_base_gesture_started
    global middle_finger_slope
    global ring_finger_slope
    global base_gesture_active
    global palm_position
    global last_palm_position
    global palm_gesture_radius_counter
    
    # Input for mouse movement
    disp_scaler = 2
    disp_offset = 0
    thumb_x = landmarks[4][0] * disp_scaler + disp_offset
    thumb_y = landmarks[4][1] * disp_scaler + disp_offset
    index_x = landmarks[8][0] * disp_scaler + disp_offset  # 8 is index finger, 12 is middle finger
    index_y = landmarks[8][1] * disp_scaler + disp_offset
    middle_x = landmarks[12][0] * disp_scaler + disp_offset  # 12 is middle finger
    middle_y = landmarks[12][1] * disp_scaler + disp_offset

    palm_position = [ (landmarks[0][0] + landmarks[2][0] + landmarks[5][0] + landmarks[17][0]) /4,  (landmarks[0][1] + landmarks[2][1] + landmarks[5][1] + landmarks[17][1]) /4 ]  # avg for the center of the palm calculated from 4 points


    # Calculate slopes of middle 2 fingers to see if the hand is in the vertical base gesture position
    try:
        middle_finger_slope = fabs((landmarks[10][0] - landmarks[11][0]) / (landmarks[10][1] - landmarks[11][1]))
        ring_finger_slope = fabs((landmarks[14][0] - landmarks[15][0]) / (landmarks[14][1] - landmarks[15][1]))
    except ZeroDivisionError:
        # If there is no slope, don't do anything
        pass

    hand_base_gesture_condition = middle_finger_slope <= middle_to_ring_threshold and ring_finger_slope <= middle_to_ring_threshold and hand_sign != 2
    
    if hand_base_gesture_condition and not hand_base_gesture_started:
        # Hand detected, start the time
        hand_base_gesture_started = True
        hand_base_gesture = time()
        print("Gesture Started")
    elif not hand_base_gesture_condition:
        # Hand not detected, reset the timer
        hand_base_gesture_started = False
        base_gesture_active = False
        print("Gesture Stopped")
    
    # When 1 second has passed, play a confirmation sound 
    if fabs(time() - hand_base_gesture) >= middle_to_ring_time and not base_gesture_active and hand_base_gesture_started:
        print("Gesture Activated")
        base_gesture_active = True
        last_palm_position = palm_position  # sets the last palm position relative to when we enter the state (non moving)
        palm_gesture_radius_counter = time()
        gesture_started.play()
        
    #   # Calculate running average of 16 values
    # if not hasattr(draw_landmarks, "landmark_history"):
    #     draw_landmarks.landmark_history = deque(maxlen=8)
    # draw_landmarks.landmark_history.append(landmark_point)
    # avg_landmark_point = np.mean(draw_landmarks.landmark_history, axis=0).astype(int).tolist()
    
    # print(avg_landmark_point)

    if hand_sign == 2:  # If hand is pointing
        mouse.position = (index_x, index_y)
    #elif hand_sign == 1:  # If hand is closed
    #    mouse.position = (index_x, index_y)

    middle_to_thumb = fabs(sqrt(pow((middle_x - thumb_x), 2) + pow((middle_y - thumb_y), 2)))

    click_condition = middle_to_thumb < current_threshold and hand_found # CHANGE THIS

    if click_condition and not mouse_pressed:  # If hand is closed and mouse not already pressed
        mouse_pressed = True
        current_threshold = click_to_off_threshold
        mouse.press(Button.left)
        print("Mouse Pressed")
        alert.play()
    elif not click_condition and mouse_pressed:  # If hand is not closed and mouse already clicked
        mouse_pressed = False
        current_threshold = click_to_on_threshold
        mouse.release(Button.left)
        print("Mouse Released")

    if base_gesture_active:
        x_change = palm_position[0] - last_palm_position[0]
        y_change = palm_position[1] - last_palm_position[1]

        # print(x_change, y_change)
        
        if x_change > palm_gesture_center_move_x_threshold: # Right swipe
            print("➡️ Right Swipe")
            swipe_action(3)
        elif x_change < -palm_gesture_center_move_x_threshold: # Left swipe
            print("⬅️ Left Swipe")
            swipe_action(2)
        elif y_change > palm_gesture_center_move_y_threshold: # Up swipe
            print("⬆️ Up Swipe")
            swipe_action(1)
        elif y_change < -palm_gesture_center_move_y_threshold: # Down swipe
            print("⬇️ Down Swipe")
            swipe_action(0)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    # parser.add_argument("--width", help='cap width', type=int, default=1920)   OLD
    # parser.add_argument("--height", help='cap height', type=int, default=1080) OLD

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    global hand_found
    
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:  # main program loop
        fps = 30
        #fps = cvFpsCalc.get()
        
        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                hand_found = True # We know that the hand was found

                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                #print(landmark_list)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )

                # Send landmarks to function
                # if 
                data_update(landmark_list, hand_sign_id)
        
        else: # If no hand is detected
            hand_found = False
            point_history.append([0, 0])
            

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)


        # landmark_list[finger_id + 1][0] - landmark_list[finger_id][0] / landmark_list[finger_id + 1][1] - landmark_list[finger_id][1]


        #print(point_history)
        # Screen reflection #############################################################

        cv.imshow('Hand Gesture Recognition', debug_image)

        ### THIS IS A JOKE. DONT DO THIS ###
        #print(debug_image)
        # UDP_IP = "192.168.0.209"
        # UDP_PORT = 5000
        # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # sock.sendto(debug_image, (UDP_IP, UDP_PORT))


    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    global palm_position

    if len(landmark_point) > 0:
        # Thumb
        #cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),(255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),(255, 255, 255), 2)

        # Index finger
        #cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),(255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),(255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),(255, 255, 255), 2)

        # Middle finger
        #cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        #cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        #cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        #cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        #cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),(0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)
        
    #cv.circle(image, (palm_position[0], palm_position[1]), 7, (0, 100, 255),-1)
    
    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # Wrist 1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # Wrist 2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # Thumb: Base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # Thumb: First joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # Thumb: Tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # Index finger: Base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # Index finger: 2nd joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # Index finger: 1st joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # Index finger: tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # Middle finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # Middle finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # Middle finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # Middle finger: tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # Ring finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # Ring finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # Ring finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # Ring finger: tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # Pinky finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # Pinky: 2nd joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # Pinky: 1st joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # Pinky: tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
        
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


def swipe_action(gesture_direction: int):
    global hand_base_gesture_started
    global base_gesture_active

    hand_base_gesture_started = False
    base_gesture_active = False

    match gesture_direction:
        case 0: # Up
            keyboard.press(Key.cmd)
            keyboard.press(Key.up)
            sleep(0.05)
            keyboard.release(Key.cmd)
            keyboard.release(Key.up)
        case 1: # Down
            keyboard.press(Key.cmd)
            keyboard.press(Key.down)
            sleep(0.05)
            keyboard.release(Key.cmd)
            keyboard.release(Key.down)
        case 2: # Left
            keyboard.press(Key.cmd)
            keyboard.press(Key.left)
            sleep(0.05)
            keyboard.release(Key.cmd)
            keyboard.release(Key.left)
        case 3: # Right
            keyboard.press(Key.cmd)
            keyboard.press(Key.right)
            sleep(0.05)
            keyboard.release(Key.cmd)
            keyboard.release(Key.right)
        case _: # No match
            print("Invalid gesture direction")

if __name__ == '__main__':
    main()
