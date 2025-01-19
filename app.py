"""
app.py

An interactive skeleton-tracking script designed for a social experiment.

Sections of this script are adapted from:
MediaPipe Pose Documentation
Hand Gesture Recognition with MediaPipe

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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

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

def main():
    cap = cv.VideoCapture(0)

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

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
            mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2) as hands:
        while cap.isOpened():

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

            cv.imshow('MediaPipe Pose with Head Model', cv.flip(white_image, 1))
            if cv.waitKey(5) & 0xFF == 27:
                break


            right_shoulder_pos = track_right_shoulder_positions(image, results_pose)
            right_elbow_pos = track_right_elbow_positions(image, results_pose)
            right_wrist_pos = track_right_wrist_positions(image, results_pose)
            left_shoulder_pos = track_left_shoulder_positions(image, results_pose)
            left_elbow_pos = track_left_elbow_positions(image, results_pose)
            left_wrist_pos = track_left_wrist_positions(image, results_pose)
            head_pos = track_head_position(image, results_pose)




            if right_shoulder_pos and right_elbow_pos and right_elbow_pos and right_wrist_pos:
                right_shoulder_elbow_slope = calc_slope(right_shoulder_pos, right_elbow_pos)
                right_elbow_wrist_slope = calc_slope(right_elbow_pos, right_wrist_pos)
                if right_count_SE == 50 and right_count_EW == 50:
                    print(f"Debug: Right shoulder pos {right_shoulder_pos}, Right elbow pos {right_elbow_pos}")
                    print(f'Right shoulder to elbow slope: {right_shoulder_elbow_slope}')
                    right_count_SE = 0

                    print(f"Debug: Right elbow pos {right_elbow_pos}, Right wrist pos {right_wrist_pos}")
                    print(f'Right elbow to wrist slope: {right_elbow_wrist_slope}')
                    right_count_EW = 0
                else:
                    if right_SE_max_slope > right_shoulder_elbow_slope > right_SE_min_slope and right_EW_max_slope > right_elbow_wrist_slope > right_EW_min_slope:
                        right_count_SE += 1
                        right_count_EW += 1
                    else:
                        right_SE_prev_slope = right_shoulder_elbow_slope
                        right_SE_max_slope = right_SE_prev_slope + .5
                        right_SE_min_slope = right_SE_prev_slope - .5
                        right_count_SE = 0

                        right_EW_prev_slope = right_elbow_wrist_slope
                        right_EW_max_slope = right_EW_prev_slope + .5
                        right_EW_min_slope = right_EW_prev_slope - .5
                        right_count_EW = 0


            if left_shoulder_pos and left_elbow_pos and left_elbow_pos and left_wrist_pos:
                left_shoulder_elbow_slope = calc_slope(left_shoulder_pos, left_elbow_pos)
                left_elbow_wrist_slope = calc_slope(left_elbow_pos, left_wrist_pos)
                if left_count_SE == 50 and left_count_EW == 50:
                    print(f"Debug: left shoulder pos {left_shoulder_pos}, left elbow pos {left_elbow_pos}")
                    print(f'left shoulder to elbow slope: {left_shoulder_elbow_slope}')
                    left_count_SE = 0

                    print(f"Debug: left elbow pos {left_elbow_pos}, left wrist pos {left_wrist_pos}")
                    print(f'left elbow to wrist slope: {left_elbow_wrist_slope}')
                    left_count_EW = 0
                else:
                    if left_SE_max_slope > left_shoulder_elbow_slope > left_SE_min_slope and left_EW_max_slope > left_elbow_wrist_slope > left_EW_min_slope:
                        left_count_SE += 1
                        left_count_EW += 1
                    else:
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

    '''         
            else:
                print("Right shoulder or elbow position data is incomplete.")

            if left_shoulder_pos and left_elbow_pos:
                print(f"Debug: Left shoulder pos {left_shoulder_pos}, Left elbow pos {left_elbow_pos}")
                left_shoulder_elbow_slope = calc_slope(left_shoulder_pos, left_elbow_pos)
                print(f'Left shoulder to elbow slope: {left_shoulder_elbow_slope}')
            else:
                print("Left shoulder or elbow position data is incomplete.")
            '''
    '''
    # Draw the pose and hand annotations on the image.
    #image.flags.writeable = True
    #image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Assume we want to overlay the image at the nose landmark (index 0 for example)
    if results_pose.pose_landmarks:
        nose_landmark = results_pose.pose_landmarks.landmark[0]
        nose_x, nose_y = int(nose_landmark.x * image.shape[1]), int(nose_landmark.y * image.shape[0])
        # Calculate position for overlay (this might need adjustment)
        overlay_pos = (nose_x - character_model.shape[1] // 2, nose_y - character_model.shape[0] // 2)
        overlay_image_alpha(white_image, character_model, overlay_pos, character_model[:, :, 3] / 255.0)


    # Flip the image horizontally for a selfie-view display.
    cv.imshow('MediaPipe Pose and Hands', cv.flip(image, 1))
    if cv.waitKey(5) & 0xFF == 27:
        break
    

    # Flip the image horizontally for a selfie-view display.
    cv.imshow('MediaPipe Pose and Hands', cv.flip(white_image, 1))
    if cv.waitKey(5) & 0xFF == 27:
        break
    
    # Check if hands are above head
    hands_above_head = check_hands_above_head(image, results_pose, results_hands)

    # Draw info text based on whether hands are above the head
    if hands_above_head:
        hand_status = "Hands Above Head"
    else:
        hand_status = "Hands Below Head"

    # Display the result on the image
    #print('hand status: ' + hand_status)
    
    # Print coordinates of pose landmarks.
            #if results_pose.pose_landmarks:
                #print_landmark_coordinates(results_pose, image.shape, type="pose")

            # Print coordinates of hand landmarks.
            #if results_hands.multi_hand_landmarks:
                #print_landmark_coordinates(results_hands, image.shape, type="hands")
                
                
    '''
    '''
            #demotration of how coords work
            hand_positions = track_hand_positions(image, results_hands)
            for hand_position in hand_positions:
                print(hand_position)
    '''

if __name__ == '__main__':
    main()
