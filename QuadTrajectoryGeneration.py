# Python Packages
import numpy as np
import random
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import cvxpy as cvx
from tqdm import tqdm

# unloaded 2D quadrotor object 
class PlanarQuadrotor:
    def __init__(self, m_Q, Iyy, d):
        self.m_Q = m_Q      # quadrotor mass
        self.Iyy = Iyy      # quadrotor second moment of inertia
        self.d = d          # length from center of mass to propellers
        self.g = 9.81       # acceleration due to gravity [m/s^2]

        # Control constraints
        self.max_thrust_per_prop = 0.75 * self.m_Q * self.g  # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = 0.


def unloaded_dynamics(unloaded_state, control, quad):
    """Continuous-time dynamics of an unloaded planar quadrotor expressed as an Euler integration."""
    
    x, z, theta, v_x, v_z, omega = unloaded_state
    T1, T2 = control
    m, Iyy, d, g = quad.m_Q, quad.Iyy, quad.d, quad.g

    ds = np.array([v_x, 
                   v_z, 
                   omega, 
                   ((T1 + T2) * np.sin(theta)) / m, 
                   ((T1 + T2) * np.cos(theta)) / m - g, 
                   (T1-T2)*d / Iyy])
    return unloaded_state + dt * ds

def linearize(fd: callable, s: np.ndarray, u: np.ndarray, dt: float, quad):
    """Explicitly linearize the unloaded dynamics around (s,u)."""
    state_dim = s.shape[1]
    control_dim = u.shape[1]
    m_Q, Iyy, d = quad.m_Q, quad.Iyy, quad.d
    A, B, c = [], [], []

    for k in range(s.shape[0]):
        s_k = s[k]
        u_k = u[k]
        x, z, theta, v_x, v_y, omega = s_k
        T1, T2 = u_k

        # derivative wrt state variables
        df_ds = np.identity(state_dim)
        df_ds[0:3,3:6] = dt*np.identity(3)
        df_ds[3,2] = dt*(T1+T2)*np.cos(theta) / m_Q
        df_ds[4,2] = -dt*(T1+T2)*np.sin(theta) / m_Q

        # derivative wrt control variables
        df_du = np.zeros((state_dim, control_dim))
        df_du[3,0] = dt*np.sin(theta) / m_Q
        df_du[3,1] = dt*np.sin(theta) / m_Q
        df_du[4,0] = dt*np.cos(theta) / m_Q
        df_du[4,1] = dt*np.cos(theta) / m_Q
        df_du[5,0] = dt*d / Iyy
        df_du[5,1] = -dt*d / Iyy

        A.append(df_ds)
        B.append(df_du)
        c.append(fd(s_k, u_k, quad) - df_ds @ s_k - df_du @ u_k)
 
    return A, B, c

# Generate time discretized control policy using Sequential Convex Programming
def generate_trajectory(fd: callable, P: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int, s_goal: np.ndarray, 
                        s0: np.ndarray, ρ: float, tol: float, max_iters: int, dt: float, quad):
    '''Solve the quadrotor trajectory problem using SCP'''
    n = Q.shape[0]    # state dimension
    m = R.shape[0]    # control dimension

    # Initialize nominal (zero control) trajectories
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = fd(s_bar[k], u_bar[k], quad)
    
    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    obj_prev = np.inf
    prog_bar = tqdm(range(max_iters))
    for i in prog_bar:
        s, u, obj = scp_iteration(fd, P, Q, R, N, s_bar, u_bar, s_goal, s0, ρ, dt, quad)
        diff_obj = np.abs(obj - obj_prev)
        prog_bar.set_postfix({'objective change': '{:.5f}'.format(diff_obj)})

        if diff_obj < tol:
            converged = True
            print('SCP converged after {} iterations.'.format(i))
            break
        else:
            obj_prev = obj
            np.copyto(s_bar, s)
            np.copyto(u_bar, u)

    if not converged:
        raise RuntimeError('SCP did not converge!')

    return s, u


def scp_iteration(fd: callable, P: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int, s_bar: np.ndarray, 
                  u_bar: np.ndarray, s_goal: np.ndarray, s0: np.ndarray, ρ: float, dt: float, quad):
    """Solve a single SCP sub-problem for the quadrotor trajectory problem."""
    A, B, c = linearize(fd, s_bar[:-1], u_bar, dt, quad)
    A, B, c = np.array(A), np.array(B), np.array(c)
    n = Q.shape[0]
    m = R.shape[0]
    T_min = quad.min_thrust_per_prop
    T_max = quad.max_thrust_per_prop
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    # ------------------- Cost -------------------
    # terminal state cost
    terminal_cost = cvx.quad_form((s_cvx[-1] - s_goal), P)

    # summed position and control cost
    sum_cost = 0
    for k in range(N):
      sum_cost += cvx.sum(cvx.quad_form((s_cvx[k]-s_goal), Q) + cvx.quad_form(u_cvx[k], R))

    objective = sum_cost + terminal_cost

    # ------------------- Constraints -------------------

    # initial position constraint
    constraints = [s_cvx[0] == s0]

    for k in range(N):
        constraints += [s_cvx[k+1] == A[k]@s_cvx[k] + B[k]@u_cvx[k] + c[k]] # linearized dynamics constraint
        constraints += [cvx.abs(u_cvx[k][0] - ((T_max+T_min)/2)) <= (T_max+T_min)/2]    # Force control bounds
        constraints += [cvx.abs(u_cvx[k][1] - ((T_max+T_min)/2)) <= (T_max+T_min)/2]    # Force control bounds

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()

    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)

    s = s_cvx.value
    u = u_cvx.value
    obj = prob.objective.value

    return s, u, obj

# --------------------------------------------------------------------------
# QUADCOPTER PARAMETERS

# Define planar quadrotor object
m_Q = 2.   # quadrotor mass (kg)
Iyy = 0.01     # moment of inertia about the out-of-plane axis (kg * m**2)
d = 0.25       # half-length (m)
quad = PlanarQuadrotor(m_Q, Iyy, d)

# --------------------------------------------------------------------------
# TRAJECTORY GENERATION

# quadrotor trajectory optimization parameters
dt = 0.01                               # Simulation time step - sec
t_final = 4.                            # Simulation run time - sec
n = 6                                   # state dimension
m = 2                                   # control dimension
s0 = np.array([0., 0., 0., 0., 0., 0.])       # initial hover state
s_goal = np.array([5., 5., 0., 0., 0., 0.])   # desired location w/ hover final velocity

# Control bounds
Tmin = quad.min_thrust_per_prop
Tmax = quad.max_thrust_per_prop

# SCP parameters
P = 150*np.eye(n)   # terminal state cost matrix
Q = np.eye(n)       # state cost matrix
R = (1/Tmax**2)*np.eye(m)  # control cost matrix
ρ = 3.              # trust region parameter
tol = 0.5          # convergence tolerance
max_iters = 100     # maximum number of SCP iterations

# Dynamics propagation
fd = unloaded_dynamics

# Solve the swing-up problem with SCP
t = np.arange(0., t_final + dt, dt)
N = t.size - 1
s_traj, u_traj = generate_trajectory(fd, P, Q, R, N, s_goal, s0, ρ, tol, max_iters, dt, quad)

# Plot and Animate quadrotor
plt.plot(t[:-1],u_traj[:,0])
plt.plot(t[:-1],u_traj[:,1])
plt.title('Control v Time')
plt.xlabel('Time (s)')
plt.ylabel('Thrust')
plt.legend(("Thrust 1","Thrust 2"))
plt.grid()
plt.show()

plt.plot(s_traj[:,0],s_traj[:,1])
plt.title('Quadrotor Trajectory')
plt.xlabel('X - position (m)')
plt.ylabel('Z - position (m)')
plt.grid()
plt.show()