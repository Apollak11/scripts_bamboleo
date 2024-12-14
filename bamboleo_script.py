#! /usr/bin/env python3

import rospy
import math
import argparse
from ultralytics import YOLO
import cv2
import math
import os
from intera_motion_interface import MotionTrajectory, MotionWaypoint, MotionWaypointOptions
from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped
from intera_interface import Limb

PI = 3.1416
########## OBJECT DETECTION#############
def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    _, ext = os.path.splitext(filename)
    return ext.lower() in video_extensions

def process_image(frame, block_model, plate_model):
    # 對圖片進行偵測
    block_results = block_model.predict(
        source=frame,
        show=False,
        save=True,
        conf=0.5
    )

    plate_results = plate_model.predict(
        source=frame,
        show=False,
        save=True,
        conf=0.5
    )

    # 解析偵測結果
    current_blocks = []
    current_plates = []

    for result in block_results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection
            block = {
                "type": block_model.names[int(class_id)],
                "position": {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                },
                "confidence": round(confidence, 2),
            }
            current_blocks.append(block)

    for result in plate_results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection
            plate = {
                "type": plate_model.names[int(class_id)],
                "position": {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                },
                "length": y2 - y1,
                "width": x2 - x1,
                "confidence": round(confidence, 2),
            }
            current_plates.append(plate)

    return current_blocks, current_plates

def draw_detections(frame, blocks, plates):
    if not plates:
        return frame

    plate = plates[0]
    circle_center_x = plate['position']['x']
    circle_center_y = plate['position']['y']
    circle_image_diameter = max(plate['length'], plate['width'])
    circle_real_diameter = 36  # cm

    for block in blocks:
        object_center_x = block['position']['x']
        object_center_y = block['position']['y']

        # Calculate the distance in the image
        image_distance = math.sqrt(
            (circle_center_x - object_center_x) ** 2 +
            (circle_center_y - object_center_y) ** 2
        )

        # Calculate the scale ratio
        scale_ratio = circle_real_diameter / circle_image_diameter

        # Calculate the real-world distance (cm)
        real_distance = image_distance * scale_ratio

        # Calculate x-axis and y-axis distances (cm)
        x_axis_image_distance = abs(circle_center_x - object_center_x)
        y_axis_image_distance = abs(circle_center_y - object_center_y)
        x_axis_real_distance = x_axis_image_distance * scale_ratio
        y_axis_real_distance = y_axis_image_distance * scale_ratio

        # Draw circles and connecting line
        cv2.circle(frame, (int(circle_center_x), int(circle_center_y)),
                   5, (0, 0, 255), -1)
        cv2.circle(frame, (int(object_center_x), int(object_center_y)),
                   5, (0, 0, 255), -1)
        cv2.line(frame, (int(circle_center_x), int(circle_center_y)),
                 (int(object_center_x), int(object_center_y)), (0, 0, 255), 2)

        # Draw arrow pointing to the x-axis direction
        x_arrow_end = (int(object_center_x), int(circle_center_y))
        cv2.arrowedLine(frame,
                        (int(circle_center_x), int(circle_center_y)),
                        x_arrow_end, (0, 255, 0), 2, tipLength=0.2)

        # Draw arrow pointing to the y-axis direction
        y_arrow_end = (int(circle_center_x), int(object_center_y))
        cv2.arrowedLine(frame,
                        (int(circle_center_x), int(circle_center_y)),
                        y_arrow_end, (255, 255, 0), 2, tipLength=0.2)

        # Set text position (middle-right offset slightly up)
        text_start_x = frame.shape[1] - 250  # 靠右側，距離右邊框 200 px
        text_start_y = frame.shape[0] // 2 - 50  # 從畫面中間往上 50 px

        # Display text on the right middle part of the image
        cv2.putText(frame, f"Distance: {real_distance:.2f} cm",
                    (text_start_x, text_start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"dx: {x_axis_real_distance:.2f} cm",
                    (text_start_x, text_start_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"dy: {y_axis_real_distance:.2f} cm",
                    (text_start_x, text_start_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame

def detect(opt):
    # Load pretrained YOLO models
    block_model = YOLO("block.pt")
    plate_model = YOLO("plate.pt")

    if opt.source == "camera":
        # Process camera type
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_blocks, current_plates = process_image(frame, block_model, plate_model)
            frame = draw_detections(frame, current_blocks, current_plates)

            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    elif is_video_file(opt.source):
        # Process video type
        cap = cv2.VideoCapture(opt.source)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = 'labeled_' + opt.source
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}")

            current_blocks, current_plates = process_image(frame, block_model, plate_model)
            frame = draw_detections(frame, current_blocks, current_plates)

            cv2.imshow("Detections", frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        print(f"Video processing is complete and has been saved to {output_path}")

    else:
        # Process image type
        frame = cv2.imread(opt.source)
        if frame is None:
            print(f"Can't read image: {opt.source}")
            return

        current_blocks, current_plates = process_image(frame, block_model, plate_model)
        frame = draw_detections(frame, current_blocks, current_plates)

        # end
        cv2.imshow("Detections", frame)
        cv2.waitKey(0)

        # save result
        output_path = 'labeled_' + opt.source
        cv2.imwrite(output_path, frame)
        print(f"Image processing is complete and has been saved to {output_path}")

    cv2.destroyAllWindows()

########### ROBOT MOVEMENT #############
### Robot moves to the following defined Position and Orientation
def move_to_position(position, orientation=None, tip_name='right_hand', timeout=None):
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

# def move_to_position_fixed_joints_5_6(position, orientation=None, tip_name='right_hand', timeout=None):
#     """
#     Bewegt den TCP zur Zielposition, während die letzten beiden Joints (5 und 6) relativ zueinander fixiert bleiben.
#
#     :param position: Zielposition des TCP (x, y, z).
#     :param orientation: (Optional) Zielorientierung des TCP (Quaternion: x, y, z, w).
#     :param tip_name: (Optional) Name des Endeffektors. Standard: 'right_hand'.
#     :param timeout: (Optional) Maximale Zeit für die Bewegung.
#     :return: True, wenn erfolgreich, sonst False.
#     """
#     limb = Limb()
#
#     # # Logge verfügbare Gelenknamen
#     # rospy.loginfo(f"Verfügbare Gelenknamen: {limb.joint_names()}")
#
#     # Aktuelle Gelenkstellungen abfragen
#     current_angles = limb.joint_angles()
#
#     # Prüfen, ob die Gelenke 'right_j5' und 'right_j6' vorhanden sind
#     if 'right_j5' not in current_angles or 'right_j6' not in current_angles:
#         rospy.logerr("Die Gelenke 'right_j5' oder 'right_j6' wurden nicht gefunden.")
#         return False
#
#     # Fixierung: Setze Joint 6 gleich Joint 5 (Winkel 0° Differenz)
#     current_angles['right_j6'] = current_angles['right_j5']
#     limb.move_to_joint_positions(current_angles)  # Synchronisiert die Gelenke
#
#     # Verwenden Sie die vorhandene `move_to_position`-Logik
#     limb = Limb()
#     traj_options = TrajectoryOptions()
#     traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
#     traj = MotionTrajectory(trajectory_options=traj_options, limb=limb)
#
#     wpt_opts = MotionWaypointOptions(max_linear_speed=0.25, max_linear_accel=0.25,
#                                      max_rotational_speed=0.5, max_rotational_accel=1.0)
#     waypoint = MotionWaypoint(options=wpt_opts.to_msg(), limb=limb)
#
#     endpoint_state = limb.tip_state(tip_name)
#     if endpoint_state is None:
#         rospy.logerr('Endpoint state not found with tip name %s', tip_name)
#         return False
#
#     pose = endpoint_state.pose
#     pose.position.x, pose.position.y, pose.position.z = position
#     if orientation:
#         pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = orientation
#
#     poseStamped = PoseStamped()
#     poseStamped.pose = pose
#     waypoint.set_cartesian_pose(poseStamped, tip_name)
#     traj.append_waypoint(waypoint.to_msg())
#
#     result = traj.send_trajectory(timeout=timeout)
#     return result and result.result

def get_current_position(tip_name='right_hand'):
    """
    Gibt die aktuelle Ist-Position und Orientierung des Roboters zurück.

    :param tip_name: Name des Endeffektors (z. B. 'right_hand').
    :return: Ein Dictionary mit Position und Orientierung oder None, falls die Position nicht abgerufen werden kann.
    """
    limb = Limb()
    endpoint_state = limb.tip_state(tip_name)

    if endpoint_state is None:
        rospy.logerr(f"Endpoint state konnte nicht für {tip_name} abgerufen werden.")
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
    Bewegt die Gelenke des Roboters zu den angegebenen Winkeln mithilfe von MotionWaypoint und MotionTrajectory.

    :param joint_angles: Dictionary mit den Gelenkwinkeln (z. B. {'right_j0': 0.0, 'right_j1': 0.5, ...}).
    :param speed_ratio: Maximales Geschwindigkeitsverhältnis (zwischen 0 und 1).
    :param accel_ratio: Maximales Beschleunigungsverhältnis (zwischen 0 und 1).
    :param timeout: Maximale Zeit (in Sekunden) für die Bewegung.
    :return: True, wenn zumindest einige Gelenke erfolgreich bewegt wurden, sonst False.
    """
    limb = Limb()
    # Erstelle MotionTrajectory und MotionWaypoint
    traj = MotionTrajectory(limb=limb)
    wpt_opts = MotionWaypointOptions(max_joint_speed_ratio=speed_ratio, max_joint_accel=accel_ratio)
    waypoint = MotionWaypoint(options=wpt_opts.to_msg(), limb=limb)

    # Setze die Ziel-Joint-Winkel
    ordered_joint_angles = [joint_angles[joint] for joint in limb.joint_names()]
    waypoint.set_joint_angles(joint_angles=ordered_joint_angles)
    traj.append_waypoint(waypoint.to_msg())

    # Führe die Trajektorie aus
    rospy.loginfo("Starte Bewegung zu den Zielwinkeln...")
    result = traj.send_trajectory(timeout=timeout)

    if result is None or not result.result:
        rospy.logerr("Fehler beim Ausführen der Trajektorie.")
        return False

    rospy.loginfo("Bewegung erfolgreich abgeschlossen.")
    return True

def get_current_joint_angles():
    """
    Gibt die aktuellen Gelenkwinkel des Roboters zurück.

    :return: Dictionary mit Gelenknamen und aktuellen Winkeln in Einheiten von Pi.
    """
    limb = Limb()
    current_angles = limb.joint_angles()
    current_angles_pi = {joint: angle / math.pi for joint, angle in current_angles.items()}
    rospy.loginfo(f"Aktuelle Gelenkwinkel (in Einheiten von Pi): {current_angles_pi}")
    return current_angles_pi

def main():
    rospy.init_node('multi_pose_execution', anonymous=True)

### Abfrage der Limbnames
    # limb = Limb()
    # limbnames = limb.joint_names()
    # rospy.loginfo(f"limb names: {limbnames}")

### Move To Waypoint template
    # waypoint = {'position': [0.4, -0.3, 0.18], 'orientation': [0.0, 1.0, 0.0, 0.0]}
    # rospy.loginfo(f"Move to waypoint with position: {waypoint['position']} and orientation: {waypoint['orientation']}" )
    # movement_success = move_to_position(waypoint['position'], waypoint['orientation'])
    # if not movement_success:
    #     rospy.logerr(f"Position of waypoint {waypoint['position']} couldn't be reached")

    # waypoint = {'position': [0.476326, -0.1525, 0.3416], 'orientation': [0.0, 0.7071, 0.7071, 0.0]}
    # rospy.loginfo(f"Move to waypoint, position: {waypoint['position']} , orientation: {waypoint['orientation']}" )
    # movement_success = move_to_position(waypoint['position'], waypoint['orientation'])
    # if not movement_success:
    #     rospy.logerr(f"Position of waypoint {waypoint['position']} couldn't be reached")

    # Ziel-Gelenkwinkel in Radiant (z. B. 0.0 entspricht 0°, 1.57 entspricht 90°)
    target_joint_angles = {
        'right_j0': 0.16*PI,  # Basisdrehung
        'right_j1': 0.11*PI,  # Schulterbewegung
        'right_j2': -0.44*PI,  # Ellbogen
        'right_j3': 0.67*PI,  # Handgelenk 1
        'right_j4': 0.61*PI,  # Handgelenk 2
        'right_j5': 0.0*PI,  # Handgelenk 3
        'right_j6': 0.05*PI # Endeffektor-Drehung
    }

    success = move_joints_to_angles(target_joint_angles)
    if success:
        rospy.loginfo("Roboter erfolgreich ausgerichtet.")
    else:
        rospy.logerr("Fehler beim Bewegen der Gelenke.")


### Get current Position
    current_pose = get_current_position()

### Aktuelle Gelenkwinkel abrufen
    current_angles = get_current_joint_angles()

if __name__ == '__main__':
    main()

