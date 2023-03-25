"""
compute_trim
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:
        12/29/2018 - RWB
        1/2022 - GND
"""

import numpy as np
from mav_sim.chap3.mav_dynamics_euler import (
    IND_EULER,
    derivatives_euler,
    euler_state_to_quat_state,
    quat_state_to_euler_state,
)
from mav_sim.chap4.mav_dynamics import forces_moments, update_velocity_data
from mav_sim.message_types.msg_delta import MsgDelta
from mav_sim.tools import types
from scipy.optimize import Bounds, minimize


def compute_trim(state0: types.DynamicState, Va: float, gamma: float, R: float = np.inf) -> tuple[types.DynamicState, MsgDelta]:
    """Compute the trim equilibrium given the airspeed and flight path angle

    Args:
        state0: An initial guess at the state
        Va: air speed
        gamma: flight path angle
        R: radius - np.inf corresponds to a straight line

    Returns:
        trim_state: The resulting trim trajectory state
        trim_input: The resulting trim trajectory inputs
    """
    # Convert the state to euler representation
    state0_euler = quat_state_to_euler_state(state0)

    # Calculate the trim
    trim_state_euler, trim_input = compute_trim_euler(state0=state0_euler, Va=Va, gamma=gamma, R=R)

    # Convert and output the returned value
    trim_state = euler_state_to_quat_state(trim_state_euler)
    return trim_state, trim_input

def compute_trim_euler(state0: types.DynamicStateEuler, Va: float, gamma: float, R: float) \
    -> tuple[types.DynamicStateEuler, MsgDelta]:
    """Compute the trim equilibrium given the airspeed, flight path angle, and radius

    Args:
        state0: An initial guess at the state
        Va: air speed
        gamma: flight path angle
        R: radius - np.inf corresponds to a straight line

    Returns:
        trim_state: The resulting trim trajectory state
        trim_input: The resulting trim trajectory inputs
    """
    # define initial state and input
    delta0 = MsgDelta(elevator=0., aileron=0., rudder=0., throttle=0.5)
    x0 = np.concatenate((state0, delta0.to_array()), axis=0)

    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                # magnitude of velocity vector is Va
                                velocity_constraint(x=x, Va_desired=Va),
                                ]),
             'jac': lambda x: np.array([
                                velocity_constraint_partial(x=x)
                                ])
             })
    # Define the bounds
    eps = 1e-12 # Small number to force equality constraint to be feasible during optimization (bug in scipy)
    lb, ub = variable_bounds(state0=state0, eps=eps)

    # solve the minimization problem to find the trim states and inputs
    psi_weight = 100000. # Weight on convergence of psi
    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(Va, gamma, R, psi_weight), bounds=Bounds(lb=lb, ub=ub),
                   constraints=cons, options={'ftol': 1e-10, 'disp': False})

    # extract trim state and input and return
    trim_state = np.array([res.x[0:12]]).T
    trim_input = MsgDelta(elevator=res.x.item(12),
                          aileron=res.x.item(13),
                          rudder=res.x.item(14),
                          throttle=res.x.item(15))
    return trim_state, trim_input

def extract_state_input(x: types.NP_MAT) -> tuple[types.NP_MAT, MsgDelta]:
    """Extracts a state vector and control message from the aggregate vector

    Args:
        x: Euler state and inputs combined into a single vector

    Returns:
        states: Euler state vector
        delta: Control command
    """
    # Extract the state and input
    state = x[0:12]
    delta = MsgDelta(elevator=x.item(12),
                     aileron=x.item(13),
                     rudder=x.item(14),
                     throttle=x.item(15))
    return state, delta

def velocity_constraint(x: types.NP_MAT, Va_desired: float) -> float:
    """Returns the squared norm of the velocity vector - Va squared

    Args:
        x: Euler state and inputs combined into a single vector
            [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r, delta_e, delta_a, delta_r, delta_t]
        Va_desired: Desired airspeed

    Returns:
        Va^2 - Va_desired^2
    """
    Va=np.sqrt(x.item(3)**2+x.item(4)**2+x.item(5)**2)
    J=Va**2-Va_desired**2

    return float(J)

def velocity_constraint_partial(x: types.NP_MAT) -> list[float]:
    """Defines the partial of the velocity constraint with respect to x

    Args:
        x: Euler state and inputs combined into a single vector
            [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r, delta_e, delta_a, delta_r, delta_t]

    Returns:
        16 element list containing the partial of the constraint wrt x
    """
    VelConPart=[0,0,0,2*x.item(3),2*x.item(4),2*x.item(5),0,0,0,0,0,0,0,0,0,0]

    return VelConPart

def variable_bounds(state0: types.DynamicStateEuler, eps: float) -> tuple[list[float], list[float]]:
    """Define the upper and lower bounds of each the states and inputs as one vector.
       If an upper and lower bound is equivalent, then the upper bound is increased by eps to
       avoid a bug in scipy. If no bound exists then +/-np.inf is used.

    Each bound will be a list of the form
        [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r, delta_e, delta_a, delta_r, delta_t]

    Args:
        state0: initial guess at the desired euler state
        eps: Small number (epsilon)

    Returns:
        lb: 16 element list defining the lower bound of each variable
        ub: 16 element list defining the upper bound of each variable
    """
    # -lower             pn                 pe                             pd
    lb = [  state0.item(IND_EULER.NORTH),   state0.item(IND_EULER.EAST),   state0.item(IND_EULER.DOWN),
        #      u     v     w        phi           theta          psi
            -np.inf, 0.,  -np.inf,  -np.pi/2,    -np.pi/2+.1,   state0.item(IND_EULER.PSI),
        #   p,  q,     r
            0., 0.,   -np.inf,
        #    \delta_e   \delta_a  \delta_r   \delta_t
            -np.pi/2,   -np.pi/2, -np.pi/2,   0.]
    # -upper             pn                       pe                             pd
    ub = [  state0.item(IND_EULER.NORTH)+eps,    state0.item(IND_EULER.EAST)+eps, state0.item(IND_EULER.DOWN)+eps,
        #      u     v          w        phi       theta          psi
             np.inf, 0.+eps,    np.inf,  np.pi/2,  np.pi/2-.1,   state0.item(IND_EULER.PSI)+eps,
        #   p,      q,       r
            0.+eps,  0.+eps, np.inf,
        #    \delta_e   \delta_a  \delta_r   \delta_t
             np.pi/2,   np.pi/2,  np.pi/2,   1.]
    return lb, ub

def trim_objective_fun(x: types.NP_MAT, Va: float, gamma: float, R: float, psi_weight: float) -> float:
    """Calculates the cost on the trim trajectory being optimized using an Euler state representation

    Objective is the norm of the desired dynamics subtract the actual dynamics (except the x-y position variables)

    Args:
        x: current state and inputs combined into a single vector
            [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r, delta_e, delta_a, delta_r, delta_t]
        Va: relative wind vector magnitude
        gamma: flight path angle
        R: radius - np.inf corresponds to a straight line

    Returns:
        J: resulting cost of the current parameters
    """
    # Extract the state and input
    matsetup=np.ones((12,))
    Q=np.diag(matsetup)
    Q[0,0]=0.
    Q[1,1]=0.
    Q[8,8]=psi_weight
    state, delta = extract_state_input(x)
    # Calculate the desired trim trajectory dynamics
    if R==np.inf:
        XDotStar=np.array(([0.],[0.],[-Va*np.sin(gamma)],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]))
    else:
        XDotStar=np.array(([0.],
                        [0.],
                        [-Va*np.sin(gamma)],
                        [0.],
                        [0.],
                        [0.],
                        [0.],
                        [0.],
                        [Va/R*np.cos(gamma)],
                        [0.],
                        [0.],
                        [0.]))


    # Calculate forces
    [VaActual,alpha,beta,_]=update_velocity_data(euler_state_to_quat_state(state))
    fm=forces_moments(euler_state_to_quat_state(state),delta,VaActual,beta,alpha)
    # Calculate the dynamics based upon the current state and input (use euler derivatives)
    XDot=derivatives_euler(state,fm)
    # Calculate the difference between the desired and actual
    Delta=XDotStar-XDot

    # Calculate the square of the difference (neglecting pn and pe)
    J=np.transpose(Delta)@Q@Delta
    J=float(J)
    return J
