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
import csv
import os

# Set up media pipeline
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

timer = 30 # Seconds before triggering ending sequence
ymca_trigger = False
# Datasheet of slopes required for a desired pose
poseData = pd.read_csv('Data\Desired_Poses - Sheet1.csv')

# Face mask for character model
character_model = cv.imread('Head.png', cv.IMREAD_UNCHANGED)

def calc_slope(first_point_position, second_point_position):
    '''
    Calculate the slope between two points in a 2D space.

    This function computes the slope between two specified points. It handles the edge case where the
    difference in the x-coordinates of the points is zero (vertical line) by returning positive or negative infinity
    depending on the direction of the line.

    Parameters:
    - first_point_position (tuple): A tuple (x, y) representing the x and y coordinates of the first point.
    - second_point_position (tuple): A tuple (x, y) representing the x and y coordinates of the second point.

    Returns:
    float: The slope of the line connecting the two points. Returns `inf` or `-inf` if the line is vertical.
    '''

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
    '''
    Extracts and returns the positions of wrists from hand landmarks detected in an image.

    This function iterates through detected hand landmarks in an image, specifically extracting the
    position of the wrist for each detected hand. The positions are normalized in the input data and
    are scaled according to the dimensions of the input image to convert them to pixel coordinates.

    Parameters:
    - image (numpy.ndarray): The image in which hands are detected. Used to scale landmark positions to pixel coordinates.
    - results_hands (mediapipe.python.solutions.hands.Hands): The results from a MediaPipe Hands model containing detected hand landmarks.

    Returns:
    list of tuple: A list of tuples, each tuple representing the (x, y) coordinates of a wrist in pixel coordinates.
    '''

    hand_positions = []
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            hand_positions.append((wrist.x * image.shape[1], wrist.y * image.shape[0]))
    return hand_positions

def track_right_shoulder_positions(image, results_pose):
    '''
    Extracts and returns the position of the right shoulder from pose landmarks detected in an image.

    This function checks for the presence of pose landmarks in the given results and specifically extracts
    the position of the right shoulder. The position is normalized in the input data and is scaled according
    to the dimensions of the input image to convert it to pixel coordinates.

    Parameters:
    - image (numpy.ndarray): The image in which the pose has been detected. Used to scale landmark positions to pixel coordinates.
    - results_pose (mediapipe.python.solutions.pose.Pose): The results from a MediaPipe Pose model containing detected pose landmarks.

    Returns:
    tuple: A tuple representing the (x, y) coordinates of the right shoulder in pixel coordinates, or an empty list if no landmarks are found.
    '''

    shoulder_positions = []
    if results_pose.pose_landmarks:
        shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_positions = (shoulder.x * image.shape[1], shoulder.y * image.shape[0])
    return shoulder_positions

def track_right_elbow_positions(image, results_pose):
    '''
    Extracts and returns the position of the right elbow from pose landmarks detected in an image.

    This function locates the right elbow landmark within the results provided by a MediaPipe Pose model.
    If the landmark is found, the function scales the normalized position to pixel coordinates based on the
    dimensions of the provided image. This allows the position to be accurately mapped within the image space.

    Parameters:
    - image (numpy.ndarray): The image in which the pose has been detected, used to convert normalized coordinates to pixel coordinates.
    - results_pose (mediapipe.python.solutions.pose.Pose): The results from a MediaPipe Pose model that includes detected pose landmarks.

    Returns:
    tuple: A tuple representing the (x, y) coordinates of the right elbow in pixel coordinates, or an empty list if no landmark is detected.
    '''

    elbow_positions = []
    if results_pose.pose_landmarks:
        elbow = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        elbow_positions = (elbow.x * image.shape[1], elbow.y * image.shape[0])
    return elbow_positions

def track_right_wrist_positions(image, results_pose):
    '''
    Extracts and returns the position of the right wrist from pose landmarks detected in an image.

    This function locates the right wrist landmark within the results provided by a MediaPipe Pose model.
    It calculates the pixel coordinates of this landmark by scaling its normalized position (relative to the image size)
    according to the dimensions of the input image. This is essential for applications that need to interact with or highlight
    specific parts of the image based on pose data.

    Parameters:
    - image (numpy.ndarray): The image where the pose has been detected, used for scaling normalized coordinates to pixel values.
    - results_pose (mediapipe.python.solutions.pose.Pose): The results from a MediaPipe Pose model containing detected pose landmarks.

    Returns:
    tuple: A tuple representing the (x, y) coordinates of the right wrist in pixel coordinates, or an empty list if no landmarks are found.
    '''

    wrist_positions = []
    if results_pose.pose_landmarks:
        wrist = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        wrist_positions = (wrist.x * image.shape[1], wrist.y * image.shape[0])
    return wrist_positions

def track_left_shoulder_positions(image, results_pose):
    '''
    Extracts and returns the position of the left shoulder from pose landmarks detected in an image.

    This function identifies the left shoulder landmark from the pose landmarks provided by a MediaPipe Pose model.
    It converts the normalized coordinates of this landmark into pixel coordinates based on the dimensions of the
    given image. This function is crucial for applications involving pose tracking where the precise location of body
    parts is required for further processing or visual output.

    Parameters:
    - image (numpy.ndarray): The image in which the pose has been detected, used to convert normalized coordinates into pixel coordinates.
    - results_pose (mediapipe.python.solutions.pose.Pose): The results from a MediaPipe Pose model that includes detected pose landmarks.

    Returns:
    tuple: A tuple representing the (x, y) coordinates of the left shoulder in pixel coordinates, or an empty list if no landmarks are found.
    '''

    shoulder_positions = []
    if results_pose.pose_landmarks:
        shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_positions = (shoulder.x * image.shape[1], shoulder.y * image.shape[0])
    return shoulder_positions

def track_left_elbow_positions(image, results_pose):
    '''
    Extracts and returns the position of the left elbow from pose landmarks detected in an image.

    This function identifies the left elbow landmark within the results provided by a MediaPipe Pose model.
    It calculates the pixel coordinates of this landmark by scaling its normalized position (relative to the image size)
    according to the dimensions of the input image. This is useful for applications that require precise interaction
    with or analysis of specific body parts in the image, such as biomechanical assessments or interactive applications.

    Parameters:
    - image (numpy.ndarray): The image in which the pose has been detected, used for scaling normalized coordinates to pixel values.
    - results_pose (mediapipe.python.solutions.pose.Pose): The results from a MediaPipe Pose model containing detected pose landmarks.

    Returns:
    tuple: A tuple representing the (x, y) coordinates of the left elbow in pixel coordinates, or an empty list if no landmarks are found.
    '''

    elbow_positions = []
    if results_pose.pose_landmarks:
        elbow = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        elbow_positions = (elbow.x * image.shape[1], elbow.y * image.shape[0])
    return elbow_positions

def track_left_wrist_positions(image, results_pose):
    '''
    Extracts and returns the position of the left wrist from pose landmarks detected in an image.

    This function identifies the left wrist landmark from the pose landmarks provided by a MediaPipe Pose model.
    It calculates the pixel coordinates of this landmark by scaling its normalized position (relative to the image size)
    according to the dimensions of the input image. This functionality is essential for applications that require detailed
    tracking of body movements or for interactive systems that respond to user gestures.

    Parameters:
    - image (numpy.ndarray): The image where the pose has been detected, used to convert normalized coordinates to pixel values.
    - results_pose (mediapipe.python.solutions.pose.Pose): The results from a MediaPipe Pose model that includes detected pose landmarks.

    Returns:
    tuple: A tuple representing the (x, y) coordinates of the left wrist in pixel coordinates, or an empty list if no landmarks are found.
    '''

    wrist_positions = []
    if results_pose.pose_landmarks:
        wrist = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        wrist_positions = (wrist.x * image.shape[1], wrist.y * image.shape[0])
    return wrist_positions

def track_head_position(image, results_pose):
    '''
    Extracts and returns the position of the head, determined by the nose landmark, from pose landmarks detected in an image.

    This function locates the nose landmark, which is used as a proxy for the head's position, within the results provided by a MediaPipe Pose model.
    The function scales the normalized coordinates of this landmark to pixel coordinates based on the dimensions of the input image. This is particularly
    useful for applications that need to track head movements or for interactive systems where head position is a control mechanism.

    Parameters:
    - image (numpy.ndarray): The image in which the pose has been detected, used for scaling normalized coordinates to pixel values.
    - results_pose (mediapipe.python.solutions.pose.Pose): The results from a MediaPipe Pose model that includes detected pose landmarks.

    Returns:
    tuple or None: A tuple representing the (x, y) coordinates of the head (nose) in pixel coordinates, or None if the landmark is not found.
    '''

    head_position = None
    if results_pose.pose_landmarks:
        nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        head_position = (nose.x * image.shape[1], nose.y * image.shape[0])
    return head_position

def check_hands_above_head(image, results_pose, results_hands):
    '''
    Determines if any hands are positioned above the head in a given image.

    This function utilizes the results from MediaPipe Pose and Hands models to compare the positions of the hands and the head.
    It first retrieves the positions of the hands and the position of the head (using the nose as a reference point) from the image.
    It then checks if the y-coordinate of any hand is less than the y-coordinate of the nose, indicating that the hand is above the head.

    Parameters:
    - image (numpy.ndarray): The image from which the positions are being determined.
    - results_pose (mediapipe.python.solutions.pose.Pose): The results of the pose detection, used to find the head's position.
    - results_hands (mediapipe.python.solutions.hands.Hands): The results of the hand detection, used to find the positions of the hands.

    Returns:
    bool: True if any hand is detected above the level of the head (nose), otherwise False.
    '''

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
    '''
    Overlays an image with transparency onto another image at a specified position.

    This function places an overlay image onto a background image at the given position, blending
    it according to an alpha mask. The alpha mask dictates the transparency of the overlay, allowing
    for smooth edges and semi-transparent effects. The function handles boundary conditions to ensure
    the overlay only affects the visible region of the background image.

    Parameters:
    - img (numpy.ndarray): The background image onto which the overlay will be placed. Must be in BGR format.
    - img_overlay (numpy.ndarray): The overlay image that will be placed on the background image. Must be in BGR format.
    - pos (tuple): A tuple (x, y) representing the top-left corner where the overlay image will be placed.
    - alpha_mask (numpy.ndarray): A 2D numpy array representing the alpha mask, which controls the transparency of the overlay.
                                  The values should be in the range [0, 1], where 0 is fully transparent and 1 is fully opaque.

    Returns:
    None: The function modifies the img array in-place with no return value.
    '''

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
    '''
    Prints the coordinates of landmarks detected in an image based on the specified type (pose or hands).

    This function processes landmark results from MediaPipe models, converting normalized landmark coordinates
    into pixel coordinates based on the image dimensions. It supports both pose landmarks (e.g., body joints)
    and hand landmarks (e.g., finger joints). The coordinates are printed directly to the console.

    Parameters:
    - results (mediapipe.python.solutions.pose.PoseLandmark or mediapipe.python.solutions.hands.HandLandmark):
      The results object containing detected landmarks. It should be the output from a MediaPipe model processing step.
    - image_shape (tuple): A tuple (height, width, channels) representing the dimensions of the image that was processed.
    - type (str, optional): The type of landmarks to process. Should be either 'pose' for body joints or 'hands' for hand joints.
      Default is 'pose'.

    Returns:
    None: This function does not return a value; it prints the landmark coordinates to the console.
    '''

    image_height, image_width, _ = image_shape
    if type == "pose" and results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            print(f'Point {idx+1}: ({landmark.x * image_width}, {landmark.y * image_height})')
    elif type == "hands" and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                print(f'Hand Point {idx+1}: ({landmark.x * image_width}, {landmark.y * image_height})')

def draw_landmarks_with_labels(image, results):
    '''
    Draws pose landmarks and their connections on an image with customized visual specifications.

    This function utilizes MediaPipe's drawing utilities to render pose landmarks on an image. It uses
    a custom drawing specification to make the landmark visualizations more prominent. The function is designed
    to visually enhance the landmarks and connections, making them easier to identify in the displayed image.

    Parameters:
    - image (numpy.ndarray): The image on which landmarks will be drawn. The image should be in RGB format.
    - results (mediapipe.python.solutions.pose.PoseLandmarks): The results object from a MediaPipe Pose model
      that contains the pose landmarks to be drawn.

    Returns:
    None: This function modifies the image in-place and does not return any value.
    '''

    if results.pose_landmarks:
        # Define custom drawing specs (thicker lines)
        custom_drawing_spec = mp_drawing.DrawingSpec(thickness=10, color=(0, 0, 0), circle_radius=0)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=custom_drawing_spec,
            connection_drawing_spec=custom_drawing_spec)

def draw_hand_landmarks_with_labels(image, results):
    '''
    Draws hand landmarks and their connections on an image using MediaPipe's default visual specifications.

    This function processes each detected hand within the given results and uses MediaPipe's drawing utilities
    to render the landmarks and their connections directly onto the provided image. The landmarks are drawn with
    the default style provided by MediaPipe, which is tailored to enhance visibility and clarity of hand joints and their interconnections.

    Parameters:
    - image (numpy.ndarray): The image on which hand landmarks will be drawn. This should be an RGB image.
    - results (mediapipe.python.solutions.hands.Hands): The results object from a MediaPipe Hands model
      that contains the hand landmarks to be drawn.

    Returns:
    None: This function modifies the image in-place and does not return any value.
    '''

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

# Simple functiion to play YMCA song upone desired pose completion
def YMCA():
    global ymca_trigger
    ymca_trigger = True
    play_sound('Data\YMCA.mp3')

def pose_check(rse_slope, rew_slope, lse_slope, lew_slope):
    '''
    Checks for specific poses based on the slope angles of right and left shoulder to elbow and elbow to wrist.

    This function uses predefined slope ranges for each segment (right shoulder to elbow, right elbow to wrist,
    left shoulder to elbow, left elbow to wrist) to filter a dataset containing potential poses. It returns the
    pose(s) that match the input slopes within a certain tolerance.

    Parameters:
    - rse_slope (float): The slope from the right shoulder to the right elbow.
    - rew_slope (float): The slope from the right elbow to the right wrist.
    - lse_slope (float): The slope from the left shoulder to the left elbow.
    - lew_slope (float): The slope from the left elbow to the left wrist.

    Returns:
    pandas.Series: A series containing the identified pose(s) based on the input slopes. If no poses match, returns an empty series.

    Note:
    - The function uses a DataFrame named 'poseData' which must be defined globally and contain the columns:
      'RSE', 'REW', 'LSE', 'LEW', and 'Pose' where each row represents a different pose and its associated slopes.
    - The tolerance for matching the slopes is set at ±1.5 units.
    '''

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
    final screen. Saves session stats to a CSV file.
    """
    print("Countdown finished! Commencing ending sequence.")
    print(f"\nTotal runtime: {round(runtime, 2)} seconds")
    print(f"Interaction time: {round(interaction_time, 2)} seconds")
    print(f"Poses hit: {poses_hit}")

    # Save the stats to a file
    save_stats(runtime, interaction_time, poses_hit)

    # Fetch global statistics
    global_stats = display_accumulated_stats()

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

    # Pause briefly on black screen
    cv.imshow('Live Feed', black_overlay)
    cv.waitKey(500)  # Display black screen for 1.5 seconds

    # Display the final white screen with stats
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(1.5, width / 800))  # Scale font size relative to screen width
    font_thickness = int(font_scale * 2)
    text_color = (0, 0, 0)  # Black text

    # Line spacing and margins dynamically adjusted to screen height and width
    margin_x = int(width * 0.05)  # 5% margin from the left
    line_spacing = int(height * 0.05)  # 5% of height for line spacing
    y_offset = int(height * 0.1)  # Start at 10% of screen height

    # Prepare local stats to display
    local_stats_text = [
        "Session Stats:",
        f"Runtime: {round(runtime, 2)} seconds",
        f"Interaction Time: {round(interaction_time, 2)} seconds",
        f"Poses Hit: {poses_hit}",
    ]

    # Prepare global stats to display
    global_stats_text = [
        "Global Stats:",
        f"Total Runtime: {round(global_stats['total_runtime'], 2)} seconds",
        f"Total Interaction Time: {round(global_stats['total_interaction_time'], 2)} seconds",
        f"Total Poses Hit: {global_stats['total_poses_hit']}",
    ]

    # Combine and display stats
    for text in local_stats_text:
        cv.putText(white_screen, text, (margin_x, y_offset), font, font_scale, text_color, font_thickness)
        y_offset += line_spacing

    # Add spacing between local and global stats
    y_offset += line_spacing

    for text in global_stats_text:
        cv.putText(white_screen, text, (margin_x, y_offset), font, font_scale, text_color, font_thickness)
        y_offset += line_spacing

    # Display the white screen with all stats
    cv.imshow('Live Feed', white_screen)
    cv.waitKey(0)  # Wait indefinitely for user input

def save_stats(runtime, interaction_time, poses_hit):
    """
    Saves the session stats (runtime, interaction time, poses hit) to a CSV file.
    If the file does not exist, it creates it with a header.
    """
    # Get the current working directory and set the /Data directory
    file_dir = os.path.join(os.getcwd(), "Data")
    
    # Ensure the directory exists
    os.makedirs(file_dir, exist_ok=True)

    # File path
    file_path = os.path.join(file_dir, "stats.csv")

    # Check if file exists
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header if the file is new
        if not file_exists:
            writer.writerow(["Total Runtime (s)", "Interaction Time (s)", "Poses Hit"])

        # Write the current session's data
        writer.writerow([round(runtime, 2), round(interaction_time, 2), poses_hit])

def display_accumulated_stats():
    """
    Display global accumulated stats from previous sessions and optionally return them.
    """
    # Get the current working directory and set the /Data directory
    file_path = os.path.join(os.getcwd(), "Data", "stats.csv")    
    total_runtime = 0
    total_interaction_time = 0
    total_poses_hit = 0

    try:
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i == 0: # Skip header
                    continue
                total_runtime += float(row[0])
                total_interaction_time += float(row[1])
                total_poses_hit += int(row[2])
    except FileNotFoundError:
        print("Stats file not found. Starting fresh.")
    
    return {
            "total_runtime": total_runtime,
            "total_interaction_time": total_interaction_time,
            "total_poses_hit": total_poses_hit,
        }

def initialize_pygame_mixer():
    pygame.mixer.init()
    # Optionally set the frequency and size of the mixer
    pygame.mixer.set_num_channels(10)  # Increase if you need more simultaneous sounds

def play_sound(file_path):
    '''
    Plays an audio file using Pygame's mixer module without blocking the main thread.

    Parameters:
    - file_path (str): The path to the audio file that needs to be played.
    '''
    sound = pygame.mixer.Sound(file_path)
    sound.play()

def main():
    '''
    Main function to capture video, detect poses and hands, and recognize specific sequences of poses.

    This function initializes a video capture device, sets up a full-screen window for displaying the video,
    and uses MediaPipe for real-time pose and hand landmark detection. It checks for specific poses (Y, M, C, A)
    based on the slopes of body segments and triggers corresponding actions, such as playing sounds. The process
    continues until the escape key is pressed or no frames are left to process.

    - Initializes video capture on a predefined camera source.
    - Sets up the application window in full-screen mode for live feed display.
    - Processes incoming video frames to detect pose and hand landmarks.
    - Calculates slopes of arm segments to detect specific poses and triggers sounds.
    - Uses a countdown mechanism to trigger a final sequence after 60 seconds of activity.

    The function handles pose detection, pose sequence recognition, and multimedia feedback based on the detection results.

    Parameters:
    None

    Returns:
    None

    Note:
    - The script is configured to work with camera ID 700; this may need adjustment based on the actual hardware.
    - It requires prior installation and proper configuration of OpenCV, MediaPipe, NumPy, and Pygame libraries.
    - Ensure all paths to sound files and other resources are correctly set relative to the script's environment.
    '''

    cap = cv.VideoCapture(700)
    initialize_pygame_mixer()  # Initialize once at the start
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

    count_pose = 1

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
            
            global ymca_trigger

            if countdown_start_time is None and ymca_trigger:  # Start countdown when YMCA is hit
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