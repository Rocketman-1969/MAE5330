"""path_follower.py implements a class for following a path with a mav
"""
from math import cos, sin

import numpy as np
from mav_sim.message_types.msg_autopilot import MsgAutopilot
from mav_sim.message_types.msg_path import MsgPath
from mav_sim.message_types.msg_state import MsgState
from mav_sim.tools.wrap import wrap


class PathFollower:
    """Class for path following
    """
    def __init__(self) -> None:
        """Initialize path following class
        """
        self.chi_inf = np.radians(50)  # approach angle for large distance from straight-line path
        self.k_path = 0.01 #0.05  # path gain for straight-line path following
        self.k_orbit = 1.# 10.0  # path gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = MsgAutopilot()  # message sent to autopilot

    def update(self, path: MsgPath, state: MsgState) -> MsgAutopilot:
        """Update the control for following the path

        Args:
            path: path to be followed
            state: current state of the mav

        Returns:
            autopilot_commands: Commands to autopilot for following the path
        """
        if path.type == 'line':
            self.autopilot_commands = follow_straight_line(path=path, state=state, k_path=self.k_path, chi_inf=self.chi_inf)
        elif path.type == 'orbit':
            self.autopilot_commands = follow_orbit(path=path, state=state, k_orbit=self.k_orbit, gravity=self.gravity)
        return self.autopilot_commands

def follow_straight_line(path: MsgPath, state: MsgState, k_path: float, chi_inf: float) -> MsgAutopilot:
    """Calculate the autopilot commands for following a straight line

    Args:
        path: straight-line path to be followed
        state: current state of the mav
        k_path: convergence gain for converging to the path
        chi_inf: Angle to take towards path when at an infinite distance from the path

    Returns:
        autopilot_commands: the commands required for executing the desired line
    """
    Pi= np.array([[state.north,state.east,-state.altitude]]).T
    epi=Pi-path.line_origin
    chi_q=np.arctan2(path.line_direction[1,0],path.line_direction[0,0])
    chi_q=wrap(chi_q,chi_q-state.chi)
    RotPI=np.array([[cos(chi_q),sin(chi_q),0],
                    [-sin(chi_q), cos(chi_q),0],
                    [0,0,1]])
    ep=RotPI@epi
    
    #corse command
    chi_c=chi_q-chi_inf*(2/np.pi)*np.arctan(k_path*ep[1,0])

    n=np.array([[path.line_direction[1,0]],[-path.line_direction[0,0]],[0]])/(np.sqrt(path.line_direction[0,0]**2+path.line_direction[1,0]**2))
    si=epi-(epi[0,0]*n[0,0]+epi[1,0]*n[1,0]+epi[2,0]*n[2,0])*n

    #altitude command
    h_c=-path.line_origin[2,0]-np.sqrt(si[0,0]**2+si[1,0]**2)*(path.line_direction[2,0]/(np.sqrt(path.line_direction[0,0]**2+path.line_direction[1,0]**2)))

    #airspeed
    Va_c=state.Va

    # Initialize the output
    autopilot_commands = MsgAutopilot()

    autopilot_commands.airspeed_command=Va_c
    autopilot_commands.altitude_command=h_c
    autopilot_commands.course_command=chi_c
    autopilot_commands.phi_feedforward=0

    # Create autopilot commands here

    return autopilot_commands


def follow_orbit(path: MsgPath, state: MsgState, k_orbit: float, gravity: float) -> MsgAutopilot:
    """Calculate the autopilot commands for following a circular path

    Args:
        path: circular orbit to be followed
        state: current state of the mav
        k_orbit: Convergence gain for reducing error to orbit
        gravity: Gravity constant

    Returns:
        autopilot_commands: the commands required for executing the desired orbit
    """
    direction=0

    if path.orbit_direction=="CCW":
        direction=-1
    else:
        direction=1
    d=np.sqrt((state.north-path.orbit_center[0,0])**2+(state.east-path.orbit_center[1,0])**2)
    phi=np.arctan2((state.east-path.orbit_center[1,0]),(state.north-path.orbit_center[0,0]))
    phi=wrap(phi,phi-state.chi)
    chi_c=phi+direction*((np.pi/2)+np.arctan(k_orbit*((d-path.orbit_radius)/path.orbit_radius)))
    h_c=-path.orbit_center[2,0]
    if (d-path.orbit_radius)/path.orbit_radius<10:
        phi_ff=direction*np.arctan2(state.Va**2,(gravity*path.orbit_radius))
    else:
        phi_ff=0

    # Initialize the output
    autopilot_commands = MsgAutopilot()
    autopilot_commands.airspeed_command=state.Va
    autopilot_commands.altitude_command=h_c
    autopilot_commands.course_command=chi_c
    autopilot_commands.phi_feedforward=phi_ff
    return autopilot_commands
