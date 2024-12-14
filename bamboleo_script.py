#! /usr/bin/env python3

import rospy
import math
from intera_motion_interface import MotionTrajectory, MotionWaypoint, MotionWaypointOptions
from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped
from intera_interface import Limb

PI = 3.1416

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

