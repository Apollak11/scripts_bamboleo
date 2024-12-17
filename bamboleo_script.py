#! /usr/bin/env python3

import rospy
import math
import argparse
import cv2
import os
import subprocess
import time
import json
import csv
from intera_motion_interface import MotionTrajectory, MotionWaypoint, MotionWaypointOptions
from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped
from intera_interface import Limb, Gripper, RobotParams
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image


PI = 3.1416
image_rotation = 0.125 # a.k.a.:j_0 rotation
default_gripper_waypoint = {'position': [0.58, -0.01, 0.4], 'orientation': [0.97915, 0.2031, -0.0024, 0.0019]}


scripts_folder = os.path.dirname(os.path.abspath(__file__))
csv_file_path = f"{scripts_folder}/block_measurement_data.csv"
txt_base_folder = f"{scripts_folder}/runs/detect"
image_folder = f"{scripts_folder}/takenImages"
bridge = CvBridge()

########## OBJECT DETECTION #############
### IMAGES
def get_next_image_name(folder):
    """
    Determines the next available filename for an image.

    :param folder: The folder where images are saved.
    :return: The path for the next image file.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Debugging: List existing files in the folder
    print(f"Checking directory: {folder}")
    print(f"Existing files in {folder}: {os.listdir(folder)}")

    # Supported image extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    # Filter files in the folder
    existing_files = [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in image_extensions and f.startswith("img")
    ]
    print(f"Filtered image files: {existing_files}")

    # Extract numerical IDs from filenames
    image_numbers = []
    for f in existing_files:
        try:
            number = int(os.path.splitext(f)[0][3:])  # Extract number after "img"
            image_numbers.append(number)
        except ValueError:
            print(f"Skipped invalid file: {f}")  # Debug invalid files

    # Debug: Print extracted numbers
    print(f"Extracted image numbers: {image_numbers}")

    # Determine the next available number
    next_number = max(image_numbers, default=0) + 1
    next_image_path = os.path.join(folder, f"img{next_number:03d}.png")

    # Debug: Print the next image name
    print(f"Next image name: {next_image_path}")

    return next_image_path

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
        # Rotate the image by 0.125 * PI counterclockwise
        rotated_img = rotate_image(cv2_img, image_rotation * PI)

        # Determine the next filename
        image_path = get_next_image_name(image_folder)

        # Save the rotated image
        cv2.imwrite(image_path, rotated_img)
        print(f"Image saved: {image_path}")
        time.sleep(5)  # so the new image gets saved appropriately before further execution of code
        rospy.signal_shutdown("Image captured and saved")

def rotate_image(image, angle_rad):
    """
    since we take the image with a rotation (because we cant set up the table otherwise)
    we need to rotate the image:
    Rotates an image counterclockwise by a given angle in radians.

    :param image: The input image (OpenCV format).
    :param angle_rad: The rotation angle in radians.
    :return: The rotated image.
    """
    # Convert angle from radians to degrees
    angle_deg = angle_rad * (180 / PI)

    # Get the image dimensions
    (h, w) = image.shape[:2]

    # Get the center of the image
    center = (w // 2, h // 2)

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    return rotated_image

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

### JSON
def get_latest_predict_folder(base_folder):
    """
    Finds the folder with the highest numeric suffix in the format 'predict', 'predict2', ..., 'predictN'.

    :param base_folder: The base directory where 'predict' folders are located.
    :return: The path to the folder with the highest number, or the base directory if no 'predict' folders exist.
    """
    # List all subdirectories in the base folder
    subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    # Filter directories starting with 'predict'
    predict_folders = [f for f in subfolders if f.startswith("predict")]

    # Extract numbers from folder names
    predict_numbers = []
    for folder in predict_folders:
        suffix = folder[7:]  # Everything after "predict"
        if suffix.isdigit():  # Check if the suffix is a number
            predict_numbers.append(int(suffix))
        elif folder == "predict":  # Handle the base 'predict' folder without a number
            predict_numbers.append(0)

    # Find the highest number
    if predict_numbers:
        max_number = max(predict_numbers)
        latest_folder = f"predict{max_number}" if max_number > 0 else "predict"
        return os.path.join(base_folder, latest_folder)
    else:
        # Return the base folder if no 'predict' folders exist
        return base_folder

def get_latest_json_file(folder_path):
    """
    Finds the latest JSON file in a folder.

    :param folder_path: Path to the folder containing JSON files.
    :return: Path to the latest JSON file or None if no file is found.
    """
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found!")
        return None

    json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    if not json_files:
        print("No JSON files found in the folder!")
        return None

    # Find the file with the latest modification timestamp
    latest_file = max(json_files, key=os.path.getmtime)
    return latest_file

def parse_detection_json(json_file):
    """
    Parses the detection information from a JSON file.

    :param json_file: Path to the JSON file.
    :return: List of parsed detected objects.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {json_file}: {e}")
        return []

    detected_objects = []
    for obj in data:
        parsed_obj = {
            "type": obj.get("type"),
            "color": obj.get("color"),
            "real_distance": obj.get("distance_information", {}).get("real_distance"),
            "x_axis_real_distance": obj.get("distance_information", {}).get("x_axis_real_distance"),
            "y_axis_real_distance": obj.get("distance_information", {}).get("y_axis_real_distance"),
            "block_width_cm": obj.get("distance_information", {}).get("block_width_cm"),
            "block_height_cm": obj.get("distance_information", {}).get("block_height_cm")
        }
        detected_objects.append(parsed_obj)

    return detected_objects

def parse_latest_detection(json_folder):
    """
    Loads the latest JSON file in the folder and returns the detected objects.

    :param json_folder: Path to the folder with JSON files.
    :return: List of detected objects.
    """
    latest_file = get_latest_json_file(json_folder)
    if latest_file is None:
        print("No valid JSON file found.")
        return []

    print(f"Latest file: {latest_file}")
    return parse_detection_json(latest_file)

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

def gripper_neutral_pos():
    #Adjust all values here
    target_joint_angles = {
        'right_j0': 0.125 * PI,  # Base rotation
        'right_j1': -0.05 * PI,  # Shoulder movement
        'right_j2': -0.5 * PI,  # Elbow
        'right_j3': 0.5 * PI,  # Wrist 1
        'right_j4': 0.45 * PI,  # Wrist 2
        'right_j5': 0.5 * PI,  # Wrist 3
        'right_j6': 1.05 * PI  # End effector rotation
    }

    move_joints_to_angles(target_joint_angles)

def camera_neutral_pos():
    target_joint_angles = {
        'right_j0': image_rotation * PI,  # Base rotation, global variable image_rotation
        'right_j1': -0.05 * PI,  # Shoulder movement
        'right_j2': -0.5 * PI,  # Elbow
        'right_j3': 0.5 * PI,  # Wrist 1
        'right_j4': 0.45 * PI,  # Wrist 2
        'right_j5': 0.0 * PI,  # Wrist 3
        'right_j6': 1.05 * PI  # End effector rotation
    }
    # ## Peer-Oles angles
    # target_joint_angles = {
    #     'right_j0': 0.125 * PI,  # Base rotation
    #     'right_j1': -0.0 * PI,  # Shoulder movement
    #     'right_j2': -0.5 * PI,  # Elbow
    #     'right_j3': 0.5 * PI,  # Wrist 1
    #     'right_j4': 0.5 * PI,  # Wrist 2
    #     'right_j5': 0.0 * PI,  # Wrist 3
    #     'right_j6': 1.05 * PI  # End effector rotation
    # }
    move_joints_to_angles(target_joint_angles)

def gripper_init():
    """
    Initializes and calibrates the gripper, then opens and closes it.
    """
    print('[DEBUG] ==> gripper_init')
    gripper = Gripper()
    gripper.reboot()
    time.sleep(5)
    gripper.calibrate()
    while not gripper.is_ready():
        print('.', end='', flush=True)
        time.sleep(0.1)
    if gripper.is_ready():
        print('\n[DEBUG] Closing gripper...')
        gripper.close()
        time.sleep(2)
        print('[DEBUG] Opening gripper...')
        gripper.open()
    else:
        print('[DEBUG] Gripper not ready.')

def gripper_control(action: str):
    """
    Controls the gripper (open or close).

    :param action: Action to perform ('open' or 'close').
    """
    print('[DEBUG] ==> gripper_controle')
    gripper = Gripper()
    if action == 'open':
        gripper.open()
    elif action == 'close':
        gripper.close()
    else:
        print(f'[DEBUG] Invalid action: {action}')

############### MATH #################
def add_waypoint_positions(waypoint1, waypoint2):
    """
    :Element by element addition of two vectors with length of 3
    :both inputs are dictionaries of position and orientation. only the postition gets added
    :return: List of added positions, orientation of position 1
    """
    if len(waypoint1['position']) != 3 or len(waypoint2['position']) != 3:
        raise ValueError("Both input lists need to consist of a position with exactly 3 values (x, y, z).")

    if 'orientation' not in waypoint1 or len(waypoint1['orientation']) != 4:
        raise ValueError("Waypoint1 needs to contain an orientation of 4 values (x, y, z, w).")

    # Positionen addieren
    new_position = [waypoint1['position'][i] + waypoint2['position'][i] for i in range(3)]

    # Orientierung unverändert übernehmen
    new_orientation = waypoint1['orientation']

    return {'position': new_position, 'orientation': new_orientation}


############ GAME LOGIC ##############
def parse_block_measurements(csv_file):
    """
    Parses block measurement data from a CSV file.

    :param csv_file: Path to the CSV file.
    :return: List of block measurement data.
    """
    blocks = []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                block = {
                    "type": row["Exact block type"],
                    "x": float(row["x"].replace(',', '.')),
                    "y": float(row["y"].replace(',', '.')),
                    "weight": float(row["weight (g)"].replace(',', '.'))
                }
                blocks.append(block)
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
    return blocks

def assign_detected_objects_to_blocks(detected_objects, block_measurements):
    """
    Assigns detected objects to the closest block measurement based on x and y distances.

    :param detected_objects: List of detected objects with x_axis_real_distance and y_axis_real_distance.
    :param block_measurements: List of blocks with x, y, and weight.
    :return: List of detected objects with matched block types and weights.
    """
    for obj in detected_objects:
        closest_block = None
        min_distance = float('inf')
        for block in block_measurements:
            distance = abs(obj["x_axis_real_distance"] - block["x"]) + abs(obj["y_axis_real_distance"] - block["y"])
            if distance < min_distance:
                min_distance = distance
                closest_block = block

        if closest_block:
            obj["exact_block_type"] = closest_block["type"]
            obj["weight"] = closest_block["weight"]
        else:
            obj["exact_block_type"] = "Unknown"
            obj["weight"] = 0.0
    return detected_objects

def sort_detected_objects_by_priority(detected_objects):
    """
    Sorts detected objects by grab priority (real_distance * weight).

    :param detected_objects: List of detected objects with real_distance and weight.
    :return: List of detected objects sorted by priority.
    """
    for obj in detected_objects:
        obj["grab_prio"] = obj.get("real_distance", float('inf')) * obj.get("weight", 0.0)

    sorted_objects = sorted(detected_objects, key=lambda x: x["grab_prio"])
    return sorted_objects

def move_robot_to_sorted_positions(sorted_objects, move_to_position_function):
    """
    Moves the robot to the sorted positions based on x and y values from the sorted_objects list.

    :param sorted_objects: List of sorted detected objects containing x and y axis real distances.
    :param move_to_position_function: Function to execute the robot movement.
    """
    relative_waypoint_put_down = {"position": [0.0, 0.4, 0.0], "orientation": [1, 0, 0, 0]}
    waypoint_put_down = add_waypoint_positions(default_gripper_waypoint, relative_waypoint_put_down)
    gripper = Gripper()

    for idx, obj in enumerate(sorted_objects):
        x_meters = obj.get("x_axis_real_distance", 0.0) / 100.0  # Convert cm to meters
        y_meters = obj.get("y_axis_real_distance", 0.0) / 100.0  # Convert cm to meters

        relative_waypoint = {"position": [x_meters, y_meters, 0.0], "orientation": [1, 0, 0, 0]}
        waypoint = add_waypoint_positions(default_gripper_waypoint, relative_waypoint)
        relative_waypoint_low = {"position": [x_meters, y_meters, -0.2], "orientation": [1, 0, 0, 0]}
        waypoint_low = add_waypoint_positions(default_gripper_waypoint, relative_waypoint_low)

        rospy.loginfo(f"Moving to waypoint {idx + 1}: Position: {waypoint['position']}, "
                      f"Orientation: {waypoint['orientation']}, Block Type: {obj['exact_block_type']}")
        move_to_position_function(waypoint['position'], waypoint['orientation'])
        move_to_position_function(waypoint_low['position'], waypoint_low['orientation'])
        gripper_control("close")
        move_to_position_function(waypoint['position'], waypoint['orientation'])
        move_to_position_function(waypoint_put_down['position'], waypoint_put_down['orientation'])
        gripper_control("open")
        move_to_position_function(default_gripper_waypoint['position'], default_gripper_waypoint['orientation'])

        rospy.loginfo(f"Successfully reached waypoint {idx + 1}")

    rospy.loginfo("Success (Maybe): Moved robot to all sorted positions")

def main():
    rospy.init_node('multi_pose_execution', anonymous=True)


###############################################
############## Initialization #################
###############################################


    camera_neutral_pos()
    gripper_init()
    # gripper_neutral_pos()

    # Get the current position of the robot
    get_current_position()

    # Fetch the current joint angles of the robot
    get_current_joint_angles()

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

    txt_folder = get_latest_predict_folder(txt_base_folder)
    rospy.loginfo(f"Analysis complete. Results are located in the folder {txt_folder}.")

    # Find the latest generated TXT file with detection results
    detected_objects = parse_latest_detection(txt_folder)
    # Output the detected objects
    for obj in detected_objects:
        rospy.loginfo(
            f"Object Type: {obj['type']}, Color: {obj['color']}, Real Distance: {obj['real_distance']:.2f} m, "
            f"X Real Distance: {obj['x_axis_real_distance']:.2f} m, Y Real Distance: {obj['y_axis_real_distance']:.2f} m"
        )


    # Parse block measurements from the CSV file
    block_measurements = parse_block_measurements(csv_file_path)

    # Assign detected objects to the closest block measurements
    matched_objects = assign_detected_objects_to_blocks(detected_objects, block_measurements)

    # Sort detected objects by grab priority
    sorted_objects = sort_detected_objects_by_priority(matched_objects)
    for obj in sorted_objects:
        rospy.loginfo(
            f"Object Type: {obj['type']}, Color: {obj['color']}, Real Distance: {obj['real_distance']:.2f} m, "
            f"X Real Distance: {obj['x_axis_real_distance']:.2f} m, Y Real Distance: {obj['y_axis_real_distance']:.2f} m, "
            f"Block Type: {obj['exact_block_type']}, Weight: {obj['weight']:.2f} g, Priority: {obj['grab_prio']:.2f}"
        )



###############################################
################## Gameplay ###################
###############################################

    # # move gripper to middle of board:
    # relative_waypoint = { 'position': [0.0, 0.0, 0.0], 'orientation': [1, 0, 0, 0]}
    # waypoint = add_waypoint_positions(default_gripper_waypoint, relative_waypoint)
    #
    # #Only to Debug:
    # rospy.loginfo(f"Move to waypoint, position: {waypoint['position']}, orientation: {waypoint['orientation']}")
    # movement_success = move_to_position(waypoint['position'], waypoint['orientation'])
    # if not movement_success:
    #     rospy.logerr(f"Could not reach waypoint position: {waypoint['position']}")

    # Move robot to the sorted positions
    move_robot_to_sorted_positions(sorted_objects, move_to_position)

    camera_neutral_pos()

if __name__ == '__main__':
    main()