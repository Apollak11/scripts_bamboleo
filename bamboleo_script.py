#! /usr/bin/env python3

import rospy
import math
import argparse
import cv2
import os
import subprocess
from intera_motion_interface import MotionTrajectory, MotionWaypoint, MotionWaypointOptions
from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped
from intera_interface import Limb
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image

PI = 3.1416
scripts_folder = "./src/intera_sdk/intera_examples/scripts"
txt_folder = f"{scripts_folder}/runs/detect/predict"
image_folder = f"{scripts_folder}/takenImages"
bridge = CvBridge()

########## OBJECT DETECTION #############

def get_next_image_name(folder):
    """
    Determines the next available filename for an image.

    :param folder: The folder where images are saved.
    :return: The path for the next image file.
    """
    # Check all files in the folder and filter by image formats
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    existing_files = [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    # Extract numerical IDs from filenames
    image_numbers = [
        int(os.path.splitext(f)[0][3:]) for f in existing_files if f.startswith("img") and f[3:].isdigit()
    ]

    # Determine the next available number
    next_number = max(image_numbers, default=0) + 1
    return os.path.join(folder, f"img{next_number:03d}.png")  # Number with leading zeros

def take_image(msg):
    """
    Callback function to receive and save an image.

    :param msg: The ROS image message.
    """

    os.makedirs(image_folder, exist_ok=True)  # Create folder if it doesn't exist

    print("Received an image!")
    try:
        # Convert ROS Image Message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        print(f"Error converting ROS image: {e}")
    else:
        # Determine the next filename
        image_path = get_next_image_name(image_folder)

        # Save the image
        cv2.imwrite(image_path, cv2_img)
        print(f"Image saved: {image_path}")
        rospy.signal_shutdown("Image captured and saved")

def parse_detection_txt(txt_path):
    """
    Reads a YOLO TXT file and returns a list of detected objects.

    :param txt_path: Path to the TXT file.
    :return: List of objects with properties.
    """
    objects = []

    if not os.path.exists(txt_path):
        print(f"File {txt_path} not found!")
        return objects

    with open(txt_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            data = line.strip().split()  # Assumes data is space-separated
            if len(data) < 6:
                print(f"Invalid line: {line}")
                continue

            class_id, x_center, y_center, width, height, confidence = map(float, data[:6])

            # Assemble object properties
            obj = {
                "id": int(class_id),  # Class ID as integer
                "type": f"Type_{int(class_id)}",  # Class name (replace with actual names if available)
                "position": {
                    "x": x_center,
                    "y": y_center
                },
                "size": {
                    "width": width,
                    "height": height
                },
                "confidence": confidence
            }
            objects.append(obj)

    return objects

def get_latest_txt_file(folder_path):
    """
    Finds the latest TXT file in a folder.

    :param folder_path: Path to the folder containing TXT files.
    :return: Path to the latest TXT file or None if no file is found.
    """
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found!")
        return None

    txt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not txt_files:
        print("No TXT files found in the folder!")
        return None

    # Find the file with the latest modification timestamp
    latest_file = max(txt_files, key=os.path.getmtime)
    return latest_file

def parse_latest_detection(txt_folder):
    """
    Loads the latest TXT file in the folder and returns the detected objects.

    :param txt_folder: Path to the folder with TXT files.
    :return: List of detected objects.
    """
    latest_file = get_latest_txt_file(txt_folder)
    if latest_file is None:
        print("No valid TXT file found.")
        return []

    print(f"Latest file: {latest_file}")
    return parse_detection_txt(latest_file)

def get_latest_image(folder_path):
    """
    Finds the latest image file in a folder.

    :param folder_path: Path to the folder with images.
    :return: Path to the latest image file or None if no file is found.
    """
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found!")
        return None

    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print("No image files found in the folder!")
        return None

    # Find the file with the latest modification timestamp
    latest_file = max(image_files, key=os.path.getmtime)
    return latest_file

########### ROBOT MOVEMENT #############
### Robot moves to the following defined Position and Orientation
def move_to_position(position, orientation=None, tip_name='right_hand', timeout=None):
    """
    Moves the robot to a specified position and orientation.

    :param position: Target position as [x, y, z].
    :param orientation: Target orientation as [x, y, z, w] (optional).
    :param tip_name: Name of the end effector (default: 'right_hand').
    :param timeout: Timeout for the movement (optional).
    :return: True if the movement was successful, False otherwise.
    """
    limb = Limb()
    traj_options = TrajectoryOptions()
    traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
    traj = MotionTrajectory(trajectory_options=traj_options, limb=limb)

    wpt_opts = MotionWaypointOptions(max_linear_speed=0.25, max_linear_accel=0.25,
                                     max_rotational_speed=0.5, max_rotational_accel=1.0)
    waypoint = MotionWaypoint(options=wpt_opts.to_msg(), limb=limb)

    endpoint_state = limb.tip_state(tip_name)
    if endpoint_state is None:
        rospy.logerr('Endpoint state not found with tip name %s', tip_name)
        return False

    pose = endpoint_state.pose
    pose.position.x, pose.position.y, pose.position.z = position
    if orientation:
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = orientation

    poseStamped = PoseStamped()
    poseStamped.pose = pose
    waypoint.set_cartesian_pose(poseStamped, tip_name)
    traj.append_waypoint(waypoint.to_msg())

    result = traj.send_trajectory(timeout=timeout)
    return result and result.result

def get_current_position(tip_name='right_hand'):
    """
    Returns the robot's current position and orientation.

    :param tip_name: Name of the end effector (e.g., 'right_hand').
    :return: A dictionary with position and orientation or None if the position could not be retrieved.
    """
    limb = Limb()
    endpoint_state = limb.tip_state(tip_name)

    if endpoint_state is None:
        rospy.logerr(f"Endpoint state could not be retrieved for {tip_name}.")
        return None

    pose = endpoint_state.pose

    current_position = {
        'position': {
            'x': pose.position.x,
            'y': pose.position.y,
            'z': pose.position.z
        },
        'orientation': {
            'x': pose.orientation.x,
            'y': pose.orientation.y,
            'z': pose.orientation.z,
            'w': pose.orientation.w
        }
    }

    rospy.loginfo(f"Current Position is: {current_position['position']}")
    rospy.loginfo(f"Current Orientation is: {current_position['orientation']}")

    return current_position

def move_joints_to_angles(joint_angles, speed_ratio=0.5, accel_ratio=0.5, timeout=None):
    """
    Moves the robot's joints to the specified angles using MotionWaypoint and MotionTrajectory.

    :param joint_angles: Dictionary with joint angles (e.g., {'right_j0': 0.0, 'right_j1': 0.5, ...}).
    :param speed_ratio: Maximum speed ratio (between 0 and 1).
    :param accel_ratio: Maximum acceleration ratio (between 0 and 1).
    :param timeout: Maximum time (in seconds) for the movement.
    :return: True if the movement was successful, False otherwise.
    """
    limb = Limb()
    traj = MotionTrajectory(limb=limb)
    wpt_opts = MotionWaypointOptions(max_joint_speed_ratio=speed_ratio, max_joint_accel=accel_ratio)
    waypoint = MotionWaypoint(options=wpt_opts.to_msg(), limb=limb)

    # Set target joint angles
    ordered_joint_angles = [joint_angles[joint] for joint in limb.joint_names()]
    waypoint.set_joint_angles(joint_angles=ordered_joint_angles)
    traj.append_waypoint(waypoint.to_msg())

    rospy.loginfo("Starting movement to target joint angles...")
    result = traj.send_trajectory(timeout=timeout)

    if result is None or not result.result:
        rospy.logerr("Error executing trajectory.")
        return False

    rospy.loginfo("Movement successfully completed.")
    return True

def get_current_joint_angles():
    """
    Returns the robot's current joint angles.

    :return: Dictionary with joint names and current angles in units of Pi.
    """
    limb = Limb()
    current_angles = limb.joint_angles()
    current_angles_pi = {joint: angle / math.pi for joint, angle in current_angles.items()}
    rospy.loginfo(f"Current joint angles (in units of Pi): {current_angles_pi}")
    return current_angles_pi

def main():
    rospy.init_node('multi_pose_execution', anonymous=True)

###############################################
############## Initialization #################
###############################################

    # Example query of limb names
    # limb = Limb()
    # limbnames = limb.joint_names()
    # rospy.loginfo(f"Limb names: {limbnames}")

    # Move to waypoint template (example position and orientation)
    # waypoint = {'position': [0.4, -0.3, 0.18], 'orientation': [0.0, 1.0, 0.0, 0.0]}
    # rospy.loginfo(f"Move to waypoint with position: {waypoint['position']} and orientation: {waypoint['orientation']}")
    # movement_success = move_to_position(waypoint['position'], waypoint['orientation'])
    # if not movement_success:
    #     rospy.logerr(f"Could not reach waypoint position: {waypoint['position']}")

    # Another waypoint example
    # waypoint = {'position': [0.476326, -0.1525, 0.3416], 'orientation': [0.0, 0.7071, 0.7071, 0.0]}
    # rospy.loginfo(f"Move to waypoint, position: {waypoint['position']}, orientation: {waypoint['orientation']}")
    # movement_success = move_to_position(waypoint['position'], waypoint['orientation'])
    # if not movement_success:
    #     rospy.logerr(f"Could not reach waypoint position: {waypoint['position']}")

    # Example joint angles (in radians, 0.0 corresponds to 0°, 1.57 corresponds to 90°)
    target_joint_angles = {
        'right_j0': 0.16 * PI,  # Base rotation
        'right_j1': 0.11 * PI,  # Shoulder movement
        'right_j2': -0.44 * PI,  # Elbow
        'right_j3': 0.67 * PI,  # Wrist 1
        'right_j4': 0.61 * PI,  # Wrist 2
        'right_j5': 0.0 * PI,   # Wrist 3
        'right_j6': 0.05 * PI   # End effector rotation
    }

    success = move_joints_to_angles(target_joint_angles)
    if success:
        rospy.loginfo("Robot successfully positioned.")
    else:
        rospy.logerr("Error positioning the robot.")

    # Get the current position of the robot
    current_pose = get_current_position()

    # Fetch the current joint angles of the robot
    current_angles = get_current_joint_angles()

###############################################
################# Get Image ###################
###############################################

    # Define the image topic
    image_topic = "/io/internal_camera/right_hand_camera/image_raw"

    # Set up the subscriber for the image topic with the callback
    rospy.Subscriber(image_topic, Image, take_image)
    print("Waiting for an image...")

    # Keep spinning until shutdown (e.g., manually with Ctrl+C)
    rospy.spin()

###############################################
################# Detection ###################
###############################################

    # Find the latest image to analyze

    latest_image = get_latest_image(image_folder)
    if latest_image is None:
        rospy.logerr("No valid image found!")
        return

    rospy.loginfo(f"Analyzing the latest image: {latest_image}")

    # Run detect.py with the latest image
    try:
        subprocess.run(
            ["python3", f"{scripts_folder}/detect.py", "--source", latest_image],
            check=True
        )
    except subprocess.CalledProcessError as e:
        rospy.logerr(f"Error running detect.py: {e}")
        return

    rospy.loginfo("Analysis complete. Results are located in the folder 'runs/detect/predict'.")

    # Find the latest generated TXT file with detection results
    detected_objects = parse_latest_detection(txt_folder)
    rospy.loginfo(f"Detected objects: {detected_objects}")

    # Output the detected objects
    for obj in detected_objects:
        rospy.loginfo(
            f"Object ID: {obj['id']}, Type: {obj['type']}, Position: {obj['position']}, Confidence: {obj['confidence']}"
        )

###############################################
################## Gameplay ###################
###############################################

    # Move the gripper down 90°
    # Keep the orientation constant
    # You can add specific movement commands here based on the detected objects

    # Move to specific positions and open/close the gripper
    # Example:
    # move_to_position([x, y, z], orientation)
    # Add logic for gripping or releasing objects

if __name__ == '__main__':
    main()