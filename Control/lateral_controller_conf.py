"""
Parameter config of lateral controller.
"""

# Config
ts = 0.1  # [s]
c_f = 155494.663  # [N / rad]
c_r = 155494.663  # [N / rad]
m_f = 570  # [kg]
m_r = 570  # [kg]
l_f = 1.165  # [m]
l_r = 1.165  # [m]
Iz = 1436.24  # [kg m2]
max_iteration = 150
eps = 0.01

state_size = 4

max_acceleration = 5.0  # [m / s^2]
max_steer_angle = 0.3  # [rad]
max_speed = 60  # [km/h]

