# Planar quadrotor object 
class PlanarQuadrotor:
    def __init__(self):
        
        # Simulation time step - sec
        self.dt = 0.01

        # Quadrotor parameters
        self.m_Q = 2.     # [kg], mass of the quadrotor
        self.m_p = 0.5    # [kg], mass of the payload
        self.Iyy = 0.01   # [kg*m^2], moment of inertia about the out-of-plane axis
        self.d = 0.25     # [m], length from center of mass to propellers
        self.l = 1.       # [m], pendulum length
        self.g = 9.81     # acceleration due to gravity [m/s^2]
        
        # Control constraints
        self.max_thrust_per_prop = 0.75 * self.m_Q * self.g  # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = 0.