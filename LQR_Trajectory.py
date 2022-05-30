import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from Tracking_LQR import LQR_tracking_gain
from Tracking_LQR import PlanarQuadrotor
from tqdm import trange
from IPython import display
from quad_animation import animate_2D_quad
import matplotlib.animation as animation

def unloaded_dynamics(state, control, quad):
    """Continuous-time dynamics of an unloaded planar quadrotor expressed as an Euler integration."""
    x, z, theta, v_x, v_z, omega = state
    T1, T2 = control
    m, Iyy, d, g = quad.m_Q, quad.Iyy, quad.d, quad.g
    ds = np.array([v_x, v_z, omega, ((T1 + T2) * np.sin(theta)) / m, ((T1 + T2) * np.cos(theta)) / m - g, (T1-T2)*d / Iyy])
    return state + dt * ds

# Define planar quadrotor object
quad = PlanarQuadrotor(m_Q=2., Iyy=0.01, d=0.25, m_p=0.)
# Target hover control
u_goal = np.array([(quad.m_Q*quad.g/2),(quad.m_Q*quad.g/2)])

# time parameters
dt = 0.01                           # Simulation time step - sec

# Waypoints for unloaded trajectory (box movement)
'''
waypoints = np.array([[3., 0., 0., 0., 0., 0.],
                      [3., 3., 0., 0., 0., 0.],
                      [0., 3., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0.]])
                      '''
waypoints = np.array([[5., 5., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0.]])

num_wp = waypoints.shape[0]
wp_dt = 400
num_steps = num_wp * wp_dt
ts = np.arange(num_steps) * dt

goal = np.zeros((num_steps, 6))
K = np.zeros((num_steps,2,6))
for i in range(num_wp):
    goal[i*wp_dt:(i+1)*wp_dt] = waypoints[i]
    K[i*wp_dt:(i+1)*wp_dt] = LQR_tracking_gain(waypoints[i], dt, quad)

states = np.zeros((num_steps, 6))
controls = np.zeros((num_steps, 2))

for i in trange(1, num_steps):
    state = states[i-1]
    delta_s = state - goal[i-1]
    controls[i-1] = u_goal + K[i-1] @ delta_s
    states[i] = unloaded_dynamics(states[i-1], controls[i-1], quad)

# Plot quadcopter trajectory
plt.plot(states[:,0],states[:,1])
plt.title('Quadrotor Trajectory')
plt.xlabel('X - position (m)')
plt.ylabel('Z - position (m)')
plt.show()

# Plot quadcopter control
plt.plot(ts,controls[:,0])
plt.plot(ts,controls[:,1])
plt.title('Control v Time')
plt.xlabel('Time (s)')
plt.ylabel('Thrust')
plt.show()

fig, anim = animate_2D_quad(ts, states, full_system=0, frameDelay=1)
#display.display(display.HTML(anim.to_html5_video()))
f = r"c://Users/alexe/Desktop/planar_quad_2.gif"
writergif = animation.PillowWriter(fps=30)
writervideo = animation.FFMpegWriter(fps=60)
anim.save(f, writer=writergif)
plt.show()