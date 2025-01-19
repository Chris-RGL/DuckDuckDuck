"""
app.py

An interactive skeleton-tracking script designed for a social experiment.

Sections of this script are adapted from:
1. [MediaPipe Pose Documentation](https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md)
2. [Hand Gesture Recognition with MediaPipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe)

Refer to the README for detailed attribution and license information.

Authors:
Christopher Gallinger-Long
Angel Rivera
Levi Salgado

Date: 01/19/2025
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import pandas as pd
from playsound import playsound
import pygame

# Set up media pipeline
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

timer = 60 # Seconds before triggering ending sequence
ymca_trigger = False
# Datasheet of slopes required for a desired pose
poseData = pd.read_csv('Data\Desired_Poses - Sheet1.csv')

# Face mask for character model
character_model = cv.imread('Head.png', cv.IMREAD_UNCHANGED)

def calc_slope(first_point_position, second_point_position):
    # Adding a small epsilon value to prevent division by zero
    epsilon = 1e-6
    dx = second_point_position[0] - first_point_position[0]
    dy = second_point_position[1] - first_point_position[1]

    # Check if the difference in x is very small to avoid division by zero
    if abs(dx) < epsilon:
        # Return a very large value to represent an infinite slope (vertical line)
        return float('inf') if dy > 0 else float('-inf')
    else:
        slope = dy / dx
    return slope

def track_hand_positions(image, results_hands):
    hand_positions = []
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            hand_positions.append((wrist.x * image.shape[1], wrist.y * image.shape[0]))
    return hand_positions

def track_right_shoulder_positions(image, results_pose):
    shoulder_positions = []
    if results_pose.pose_landmarks:
        shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_positions = (shoulder.x * image.shape[1], shoulder.y * image.shape[0])
    return shoulder_positions

def track_right_elbow_positions(image, results_pose):
    elbow_positions = []
    if results_pose.pose_landmarks:
        elbow = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        elbow_positions = (elbow.x * image.shape[1], elbow.y * image.shape[0])
    return elbow_positions

def track_right_wrist_positions(image, results_pose):
    wrist_positions = []
    if results_pose.pose_landmarks:
        wrist = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        wrist_positions = (wrist.x * image.shape[1], wrist.y * image.shape[0])
    return wrist_positions

def track_left_shoulder_positions(image, results_pose):
    shoulder_positions = []
    if results_pose.pose_landmarks:
        shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_positions = (shoulder.x * image.shape[1], shoulder.y * image.shape[0])
    return shoulder_positions

def track_left_elbow_positions(image, results_pose):
    elbow_positions = []
    if results_pose.pose_landmarks:
        elbow = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        elbow_positions = (elbow.x * image.shape[1], elbow.y * image.shape[0])
    return elbow_positions

def track_left_wrist_positions(image, results_pose):
    wrist_positions = []
    if results_pose.pose_landmarks:
        wrist = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        wrist_positions = (wrist.x * image.shape[1], wrist.y * image.shape[0])
    return wrist_positions

def track_head_position(image, results_pose):
    head_position = None
    if results_pose.pose_landmarks:
        nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        head_position = (nose.x * image.shape[1], nose.y * image.shape[0])
    return head_position

def check_hands_above_head(image, results_pose, results_hands):
    hand_positions = track_hand_positions(image, results_hands)
    head_position = track_head_position(image, results_pose)

    hands_above_head = False

    if hand_positions and head_position:
        head_y = head_position[1]  # Nose Y-coordinate
        for hand_pos in hand_positions:
            hand_y = hand_pos[1]  # Wrist Y-coordinate
            if hand_y < head_y:  # Hands above head if wrist is above nose
                hands_above_head = True

    return hands_above_head

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return
    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha
    for c in range(img.shape[2]):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

def print_landmark_coordinates(results, image_shape, type="pose"):
    image_height, image_width, _ = image_shape
    if type == "pose" and results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            print(f'Point {idx+1}: ({landmark.x * image_width}, {landmark.y * image_height})')
    elif type == "hands" and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                print(f'Hand Point {idx+1}: ({landmark.x * image_width}, {landmark.y * image_height})')

def draw_landmarks_with_labels(image, results):
    if results.pose_landmarks:
        # Define custom drawing specs (thicker lines)
        custom_drawing_spec = mp_drawing.DrawingSpec(thickness=10, color=(0, 0, 0), circle_radius=0)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=custom_drawing_spec,
            connection_drawing_spec=custom_drawing_spec)
        '''
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, image.shape[1], image.shape[0])
            if landmark_px:
                cv.circle(image, landmark_px, 5, (0, 255, 0), -1)
                cv.putText(image, str(idx+1), (landmark_px[0] + 5, landmark_px[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        '''

def draw_hand_landmarks_with_labels(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            '''
            for idx, landmark in enumerate(hand_landmarks.landmark):
                landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, image.shape[1], image.shape[0])
                if landmark_px:
                    cv.circle(image, landmark_px, 5, (0, 255, 0), -1)
                    cv.putText(image, f"{idx+1}", (landmark_px[0] + 5, landmark_px[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            '''
def YMCA():
    ymca_trigger = True
    play_sound('Data\YMCA.mp3')

def pose_check(rse_slope, rew_slope, lse_slope, lew_slope):
    #angle parameters
    rse_max = rse_slope + 1.5
    rse_min = rse_slope - 1.5
    rew_max = rew_slope + 1.5
    rew_min = rew_slope - 1.5
    lse_max = lse_slope + 1.5
    lse_min = lse_slope - 1.5
    lew_max = lew_slope + 1.5
    lew_min = lew_slope - 1.5

    filtered_data = poseData[(poseData['RSE'] < rse_max) & (poseData['RSE'] > rse_min)]
    if len(filtered_data) > 0:
        filtered_data = filtered_data[(filtered_data['REW'] < rew_max) & (filtered_data['REW'] > rew_min)]
        if len(filtered_data) > 0:
            filtered_data = filtered_data[(filtered_data['LSE'] < lse_max) & (filtered_data['LSE'] > lse_min)]
            if len(filtered_data) > 0:
                filtered_data = filtered_data[(filtered_data['LEW'] < lew_max) & (filtered_data['LEW'] > lew_min)]
    if not filtered_data.empty:
        return filtered_data['Pose']
    else:
        return pd.Series()  # Return an empty series if no match is found


def ending_sequence(image, runtime, interaction_time, poses_hit):
    """
    Implements the ending sequence by fading the current frame to
    black using the existing OpenCV window, then displays stats and
    final screen.
    """
    print("Countdown finished! Commencing ending sequence.")

    height, width, _ = image.shape
    black_overlay = np.zeros((height, width, 3), dtype=np.uint8)  # Black image
    white_screen = np.ones((height, width, 3), dtype=np.uint8) * 255  # White image

    # Total number of frames for the fade effect (assuming 30 FPS)
    total_frames = int(3 * 30)

    # Fade to black
    for i in range(total_frames):
        alpha = i / total_frames
        blended_frame = cv.addWeighted(image, 1 - alpha, black_overlay, alpha, 0)
        cv.imshow('Live Feed', blended_frame)
        if cv.waitKey(1) & 0xFF == 27:  # Break if ESC is pressed
            break
        time.sleep(1 / 30)  # Simulate 30 FPS

    # Pause briefly when fully black
    cv.imshow('Live Feed', black_overlay)
    cv.waitKey(500)

    # Fade from black to new screen
    for i in range(total_frames):
        alpha = i / total_frames
        blended_frame = cv.addWeighted(black_overlay, 1 - alpha, white_screen, alpha, 0)
        cv.imshow('Live Feed', blended_frame)
        if cv.waitKey(1) & 0xFF == 27:  # Break if ESC is pressed
            break
        time.sleep(1 / 30)  # Simulate 30 FPS

    # Display the final new screen
    cv.imshow('Live Feed', white_screen)
    cv.waitKey(0)  # Wait indefinitely for user input

def play_sound(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Check if the audio is still playing
        pygame.time.Clock().tick(10)  # Wait a little for the audio to finish

def main():
    cap = cv.VideoCapture(700)
    #cap = cv.VideoCapture(0)

    countdown_triggered = False
    countdown_start_time = None
    interaction_start_time = None
    program_start_time = time.time() # Track total runtime

    # Create a named window that can be resized
    cv.namedWindow('Live Feed', cv.WINDOW_NORMAL)
    # Set the window to full screen
    cv.setWindowProperty('Live Feed', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


    right_SE_prev_slope = 0
    right_SE_max_slope = 0
    right_SE_min_slope = 0
    right_count_SE = 0

    right_EW_prev_slope = 0
    right_EW_max_slope = 0
    right_EW_min_slope = 0
    right_count_EW = 0

    left_SE_prev_slope = 0
    left_SE_max_slope = 0
    left_SE_min_slope = 0
    left_count_SE = 0

    left_EW_prev_slope = 0
    left_EW_max_slope = 0
    left_EW_min_slope = 0
    left_count_EW = 0

    count_pose = 0

    y_pose_detected = False
    m_pose_detected = False
    c_pose_detected = False
    a_pose_detected = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
            mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2) as hands:
        while cap.isOpened():
            # Start interaction_time tracking if a pose has been hit
            if count_pose == 1:
                interaction_start_time = time.time()

            if countdown_start_time is None & ymca_trigger == True:  # Start countdown when YMCA is hit
                countdown_start_time = time.time()

            # Check if 60 seconds have elapsed since the countdown started
            if countdown_start_time:
                elapsed_time = time.time() - countdown_start_time
                if elapsed_time >= timer and not countdown_triggered:
                    interaction_time = time.time() - interaction_start_time
                    total_elapsed_time = time.time() - program_start_time # Calculate elapsed time since the program started
                    flipped_white_image = cv.flip(white_image, 1)  # Apply the same flip
                    ending_sequence(flipped_white_image, total_elapsed_time, interaction_time, count_pose)              
                    countdown_triggered = True

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the image to RGB before processing.
            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            # Process with Pose model
            results_pose = pose.process(image)
            # Process with Hands model
            results_hands = hands.process(image)


            # Create a blank white image
            white_image = np.ones_like(image) * 255
            # Convert back to BGR for displaying
            white_image = cv.cvtColor(white_image, cv.COLOR_RGB2BGR)

            draw_landmarks_with_labels(white_image, results_pose)
            draw_hand_landmarks_with_labels(white_image, results_hands)

            if results_pose.pose_landmarks:
                nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                leye = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER]
                reye = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER]

                # Calculate the distance for scaling
                eye_distance = np.linalg.norm([leye.x - reye.x, leye.y - reye.y]) * image.shape[1]
                scaling_factor = eye_distance / 200.0  # Assume 200px is the real-world eye distance at 1m

                # Calculate the position for overlay
                nose_pos = (int(nose.x * image.shape[1]), int(nose.y * image.shape[0]))

                # Scale the head model
                scaled_head = cv.resize(character_model, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_LINEAR)
                alpha_mask = scaled_head[:, :, 3] / 255.0
                overlay_pos = (nose_pos[0] - scaled_head.shape[1] // 2, nose_pos[1] - scaled_head.shape[0] // 2)

                overlay_image_alpha(white_image, scaled_head[:, :, :3], overlay_pos, alpha_mask)

            cv.imshow('Live Feed', cv.flip(white_image, 1))
            if cv.waitKey(5) & 0xFF == 27:
                break


            right_shoulder_pos = track_right_shoulder_positions(image, results_pose)
            right_elbow_pos = track_right_elbow_positions(image, results_pose)
            right_wrist_pos = track_right_wrist_positions(image, results_pose)
            left_shoulder_pos = track_left_shoulder_positions(image, results_pose)
            left_elbow_pos = track_left_elbow_positions(image, results_pose)
            left_wrist_pos = track_left_wrist_positions(image, results_pose)
            head_pos = track_head_position(image, results_pose)



            if right_shoulder_pos and right_elbow_pos and right_elbow_pos and right_wrist_pos and \
                    left_shoulder_pos and left_elbow_pos and left_elbow_pos and left_wrist_pos:
                right_shoulder_elbow_slope = calc_slope(right_shoulder_pos, right_elbow_pos)
                right_elbow_wrist_slope = calc_slope(right_elbow_pos, right_wrist_pos)
                left_shoulder_elbow_slope = calc_slope(left_shoulder_pos, left_elbow_pos)
                left_elbow_wrist_slope = calc_slope(left_elbow_pos, left_wrist_pos)
                if right_count_SE == 30 and right_count_EW == 30 and left_count_SE == 30 and left_count_EW == 30:
                    if y_pose_detected == True and m_pose_detected == True and c_pose_detected == True and a_pose_detected == True:
                        YMCA()
                        y_pose_detected = False
                        m_pose_detected = False
                        c_pose_detected = False
                        a_pose_detected = False
                    struck_pose = pose_check(right_shoulder_elbow_slope, right_elbow_wrist_slope, left_shoulder_elbow_slope, left_elbow_wrist_slope)
                    if len(struck_pose) > 1:
                        print('Too many pose matches')
                    else:
                        if not struck_pose.empty:
                            match struck_pose.iloc[0]:  # Use iloc to safely access the first element
                                case 'Y':
                                    if not y_pose_detected:
                                        print('Y!')
                                        play_sound('Data\\blip.mp3')
                                        count_pose += 1
                                    y_pose_detected = True
                                case 'M':
                                    if not m_pose_detected:
                                        print('M!')
                                        play_sound('Data\\blip.mp3')
                                        count_pose += 1
                                    m_pose_detected = True
                                case 'C':
                                    if not c_pose_detected:
                                        print('C!')
                                        play_sound('Data\\blip.mp3')
                                        count_pose += 1
                                    c_pose_detected = True
                                case 'A':
                                    if not a_pose_detected:
                                        print('A!')
                                        play_sound('Data\\blip.mp3')
                                        count_pose += 1
                                    a_pose_detected = True

                else:
                    if right_SE_max_slope > right_shoulder_elbow_slope > right_SE_min_slope and right_EW_max_slope > right_elbow_wrist_slope > right_EW_min_slope and \
                            left_SE_max_slope > left_shoulder_elbow_slope > left_SE_min_slope and left_EW_max_slope > left_elbow_wrist_slope > left_EW_min_slope:
                        right_count_SE += 1
                        right_count_EW += 1
                        left_count_SE += 1
                        left_count_EW += 1
                    else:
                        right_SE_prev_slope = right_shoulder_elbow_slope
                        right_SE_max_slope = right_SE_prev_slope + .5
                        right_SE_min_slope = right_SE_prev_slope - .5
                        right_count_SE = 0

                        right_EW_prev_slope = right_elbow_wrist_slope
                        right_EW_max_slope = right_EW_prev_slope + .5
                        right_EW_min_slope = right_EW_prev_slope - .5
                        right_count_EW = 0

                        left_SE_prev_slope = left_shoulder_elbow_slope
                        left_SE_max_slope = left_SE_prev_slope + .5
                        left_SE_min_slope = left_SE_prev_slope - .5
                        left_count_SE = 0

                        left_EW_prev_slope = left_elbow_wrist_slope
                        left_EW_max_slope = left_EW_prev_slope + .5
                        left_EW_min_slope = left_EW_prev_slope - .5
                        left_count_EW = 0




    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()