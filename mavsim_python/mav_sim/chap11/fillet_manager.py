"""Provides an implementation of the fillet path manager for waypoint following as described in
   Chapter 11 Algorithm 8
"""
from typing import cast

import numpy as np
from mav_sim.chap11.path_manager_utilities import (
    EPSILON,
    HalfSpaceParams,
    WaypointIndices,
    extract_waypoints,
    get_airspeed,
    inHalfSpace,
)
from mav_sim.message_types.msg_path import MsgPath
from mav_sim.message_types.msg_state import MsgState
from mav_sim.message_types.msg_waypoints import MsgWaypoints
from mav_sim.tools.types import NP_MAT
from tomlkit import array


def fillet_manager(state: MsgState, waypoints: MsgWaypoints, ptr_prv: WaypointIndices,
                 path_prv: MsgPath, hs_prv: HalfSpaceParams, radius: float, manager_state: int) \
                -> tuple[MsgPath, HalfSpaceParams, WaypointIndices, int]:

    """Update for the fillet manager.
       Updates state machine if the MAV enters into the next halfspace.

    Args:
        state: current state of the vehicle
        waypoints: The waypoints to be followed
        ptr_prv: The indices that were being used on the previous iteration (i.e., current waypoint
                 inidices being followed when manager called)
        hs_prv: The previous halfspace being looked for (i.e., the current halfspace when manager called)
        radius: minimum radius circle for the mav
        manager_state: Integer state of the manager
                Value of 1 corresponds to following the straight line path
                Value of 2 corresponds to following the arc between straight lines

    Returns:
        path (MsgPath): Path to be followed
        hs (HalfSpaceParams): Half space parameters corresponding to the next change in state
        ptr (WaypointIndices): Indices of the current waypoint being followed
        manager_state (int): The current state of the manager

    """
    # Default the outputs to be the inputs
    path = path_prv
    hs = hs_prv
    ptr = ptr_prv

    # Insert code here
    if waypoints.flag_waypoints_changed is True: # True when waypoints are new

        waypoints.flag_waypoints_changed = False # Set to False to indicate that the waypoints have now been seen

        ptr = WaypointIndices() # Resets the pointers

        manager_state = 1 # Initialize the manager state

    if manager_state == 1:
        (path,hs)=construct_fillet_line(waypoints=waypoints,ptr=ptr,radius=radius)
        pos= np.array([[state.north],[state.east],[-state.altitude]])
        in_hs = inHalfSpace(pos=pos,hs=hs)
        if in_hs == True:
            (path,hs)=construct_fillet_circle(waypoints=waypoints,ptr=ptr,radius=radius)
            manager_state=2
    elif manager_state ==2:
        pos= np.array([[state.north],[state.east],[-state.altitude]])
        in_hs = inHalfSpace(pos=pos,hs=hs)
        if in_hs == True:
            ptr.increment_pointers(waypoints.num_waypoints)
            manager_state=1
    return (path, hs, ptr, manager_state)

def construct_fillet_line(waypoints: MsgWaypoints, ptr: WaypointIndices, radius: float) \
    -> tuple[MsgPath, HalfSpaceParams]:
    """Define the line on a fillet and a halfspace for switching to the next fillet curve.

    The line is created from the previous and current waypoints with halfspace defined for
    switching once a circle of the specified radius can be used to transition to the next line segment.

    Args:
        waypoints: The waypoints to be followed
        ptr: The indices of the waypoints being used for the path
        radius: minimum radius circle for the mav

    Returns:
        path: The straight-line path to be followed
        hs: The halfspace for switching to the next waypoint
    """
    # Extract the waypoints (w_{i-1}, w_i, w_{i+1})
    (previous, current, next_wp) = extract_waypoints(waypoints=waypoints, ptr=ptr)

    path = MsgPath()
    path.plot_updated = False
    path.airspeed= get_airspeed(waypoints, ptr)
    path.line_origin=previous
    path.line_direction=np.array(current-previous)/np.linalg.norm(current-previous)
    

    # Construct the halfspace
    hs = HalfSpaceParams()

    q_i=np.array(next_wp-current)/np.linalg.norm(next_wp-current)
    q_imin1=path.line_direction
    phi=np.arccos(-q_imin1.T@q_i)
    hs.normal=q_imin1

    if phi<EPSILON:
        hs.point=current
    else:
        hs.point=current-(radius/np.tan(phi/2))*q_imin1

    return (path, hs)

def construct_fillet_circle(waypoints: MsgWaypoints, ptr: WaypointIndices, radius: float) \
    -> tuple[MsgPath, HalfSpaceParams]:
    """Define the circle on a fillet

    Args:
        waypoints: The waypoints to be followed
        ptr: The indices of the waypoints being used for the path
        radius: minimum radius circle for the mav

    Returns:
        path: The straight-line path to be followed
        hs: The halfspace for switching to the next waypoint
    """
    # Extract the waypoints (w_{i-1}, w_i, w_{i+1})
    (previous, current, next_wp) = extract_waypoints(waypoints=waypoints, ptr=ptr)

    # Construct the path
    path = MsgPath()
    j=np.array([[0,-1,0],[1,0,0],[0,0,-1]])
    path.plot_updated = False
    path.airspeed= get_airspeed(waypoints, ptr)
    q_i=np.array(next_wp-current)/np.linalg.norm(next_wp-current)
    q_imin1=np.array(current-previous)/np.linalg.norm(current-previous)
    phi=np.arccos(-q_imin1.T@q_i)

    if np.sin(phi/2)<EPSILON:
        s_i=0
    else:
        s_i=radius/np.sin(phi/2)

    if np.linalg.norm(q_imin1-q_i)>EPSILON:
        path.orbit_center=current-(s_i)*(q_imin1-q_i)/np.linalg.norm(q_imin1-q_i)
    else:
        path.orbit_center=current+j@q_imin1*radius


    path.orbit_radius=radius
    direction=np.sign(q_imin1[0]*q_i[1]-q_imin1[1]*q_i[0])
    if direction>0:
        path.orbit_direction = "CW"
    else:
        path.orbit_direction = "CCW"
    

    # Define the switching halfspace
    hs = HalfSpaceParams()
    if phi<EPSILON:
        hs.point=current
        hs.normal=current
    else:
        hs.point=current+(radius/np.tan(phi/2))*q_i
        hs.normal=q_i
        
    path.type = "orbit"
    return (path, hs)
