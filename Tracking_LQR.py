import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# unloaded 2D quadrotor object class
class PlanarQuadrotor:
    def __init__(self, m_Q, Iyy, d, m_p = 0.):
        self.m_Q = m_Q      # quadrotor mass
        self.Iyy = Iyy      # quadrotor second moment of inertia
        self.d = d          # length from center of mass to propellers
        self.m_p = m_p      # payload mass (optional)
        self.g = 9.81       # acceleration due to gravity [m/s^2]

        # Control constraints
        self.max_thrust_per_prop = self.m_Q * self.g  # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = 0.

def Jacobians(s: np.ndarray, u: np.ndarray, dt: float, quad):
    """Calculate state and control Jacobians around (s*,u*)."""
    state_dim = s.shape[0]
    control_dim = u.shape[0]
    m, Iyy, d = quad.m_Q, quad.Iyy, quad.d
    x, z, theta, v_x, v_y, omega = s
    T1, T2 = u

    # derivative wrt state variables
    df_ds = np.identity(state_dim)
    df_ds[0:3,3:6] = dt*np.identity(3)
    df_ds[3,2] = dt*(T1+T2)*np.cos(theta) / m
    df_ds[4,2] = -dt*(T1+T2)*np.sin(theta) / m

    # derivative wrt control variables
    df_du = np.zeros((state_dim, control_dim))
    df_du[3,0] = dt*np.sin(theta) / m
    df_du[3,1] = dt*np.sin(theta) / m
    df_du[4,0] = dt*np.cos(theta) / m
    df_du[4,1] = dt*np.cos(theta) / m
    df_du[5,0] = dt*d / Iyy
    df_du[5,1] = -dt*d / Iyy

    return df_ds, df_du

# Compute deviation variable gain matrix
def LQR_tracking_gain(s_goal: np.ndarray, dt: float, quad):
    # Hover at final desired state
    u_goal = np.array([(quad.m_Q*quad.g/2),(quad.m_Q*quad.g/2)])
    # Deviation cost matrices
    Q_LQR = np.diag([1e3, 1e3, 1e3, 1e3, 1e3, 1e3])  # state deviation cost matrix
    R_LQR = 1e-3*np.eye(2)                           # control deviation cost matrix

    # Initialize cost-to-go matrix to 0
    P_inf = np.zeros_like(Q_LQR)

    A, B = Jacobians(s_goal, u_goal, dt, quad)
    # P_next stores P_{k+1} matrix
    P_next = deepcopy(P_inf)
    not_converged = True
    while not_converged:
        # Ricatti Recursion update step
        K = -1 * np.linalg.inv(R_LQR + B.T @ P_next @ B) @ B.T @ P_next @ A
        P = Q_LQR + A.T @ P_next @ (A + B @ K)

        # maximum element-wise norm condition ||P_k+1 - P_k||_max < 1e-4
        if np.all(np.absolute(P_next - P) < 1e-5):
            not_converged = False

        # Update cost-to-go matrix for next loop
        P_next = deepcopy(P)

    # Infinite horizon deviation variable gain matrix
    return K

# Generate Force and Torque control for unloaded quadrotor
def generate_control(s0: np.ndarray, s_goal: np.ndarray, K: np.ndarray, quad):

    # State deviation variable
    delta_s = s0 - s_goal

    # Control law [Thrust 1, Thrust 2]
    u_prop = K @ delta_s

    # Convert propeller [Thrust 1, Thrust 2] -> [Force, Torque]
    u = np.zeros((2,1))
    u[0] = u_prop[0] + u_prop[1]
    u[1] = (u_prop[0] - u_prop[1]) * quad.d

    return u
