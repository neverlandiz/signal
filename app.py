import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

import pyautogui

from libs.Kazuhito00.helpers import *
from model import KeyPointClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=480)
    parser.add_argument("--height", help='cap height', type=int, default=270)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", help='min_detection_confidence', type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type=int, default=0.5)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()


    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    history_length = 16
    point_history = deque(maxlen=history_length)

    finger_gesture_history = deque(maxlen=history_length)

    mode = 0

    clock = 0

    while True:
        fps = cvFpsCalc.get()

        # exit
        key = cv.waitKey(10)
        if key == 27:  
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                # gestures
                if clock % 30 == 0:
                    if hand_sign_id == 0:
                        print("open terminal")
                        pyautogui.hotkey('alt', 'enter')
                    
                    elif hand_sign_id == 1:
                        print("swipe window right")
                        pyautogui.hotkey('ctrl', 'alt', 'right')
                        
                    elif hand_sign_id == 2:
                        print("swipe window left")
                        pyautogui.hotkey('ctrl', 'alt', 'left')

                    elif hand_sign_id == 3:
                        print("esc")
                        pyautogui.hotkey('esc')
                    
                    elif hand_sign_id == 4:
                        print("rain")
                        pyautogui.typewrite('cmatrix')
                        pyautogui.hotkey('enter')
                    
                    elif hand_sign_id == 5:
                        print("paste")
                        pyautogui.hotkey('ctrl', 'v')
                    
                    elif hand_sign_id == 6:
                        print("copy")
                        pyautogui.hotkey('ctrl', 'c')
                    
                    elif hand_sign_id == 7:
                        print("open chrome")
                        pyautogui.typewrite('google-chrome')
                        pyautogui.hotkey('enter')
                    
                    elif hand_sign_id == 8:
                        print("open incognito chrome")
                        pyautogui.typewrite('google-chrome --incognito')
                        pyautogui.hotkey('enter')
                
                clock += 1

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)


                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    "",
                )

        debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()




if __name__ == '__main__':
    main()
