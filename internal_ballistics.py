from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import math
import numpy as np
import tools
from prop_cross_sections import A_b
from prop_burn_rates import prop_values
from numerical_solvers import rk4, adams_bashforth, forward_euler

# constants
R_univ = 8.314
T_a = 229 # standard temperature of air, [units: K]
p_a = 30090 # standard pressure of air, [units: Pa]
rho_a = 0.460 # standard density of air, [units: kg/m^3]
M_a_recip = 1 / 0.029 # standard molar mass reciprical of air, [units: mol/ kg]

# initial conditions
T_0 = T_a # [units: K]
c_p_c_0 = 1004 # [units: J/kg-K]
M_c_recip_0 = M_a_recip # [units: mol/kg]
V_c_0 = 3e-6 # [units: m^3}]
m_c_0 = rho_a * V_c_0 # [units: kg]
p_c_0 = p_a # [units: Pa]
x_c_0 = 0 # [units: m]
R_c_0 = R_univ * M_a_recip # [units: J/kg-K]
gamma_c_0 = 1.25 # [units: -]

# T_prop = 1600 # make T_prop = T_prop(x_c) a function of burn distance
# c_p_prop = 1200 # make c_p_prop = c_p_prop(x_c) a function of burn distance 
rho_p = 1600 # [kg/m^3]

A_t = (3.22 / 2 / 1000) ** 2 * math.pi
# A_b = (25 / 1000) ** 2 * math.pi # make A_b = A_b(x_c) a function of burn distance 
A_e = A_t * 3.8

n = 0.45 # make n = n(x_c)
# a = 9 * (1e6)**(-n) * 1e-3 # make a = a(x_c)

# M_prop_recip = M_a_recip # make M_prop_recip = M_prop_recip(x_c)

def internal_ballistics(y, t):

    T_c, c_p_c, M_c_recip, m_c, p_c, V_c, x_c, R_c, gamma_c = y

    print('x_c: ', str(x_c * 1000), ' mm')

    # print('burn_rate_coeff: ', burn_rate_coeff(x_c))

    # dx_c_dt = a * p_c ** n # 

    [a, T_prop, c_p_prop, M_prop_recip] = prop_values(x_c)

    # print(prop_values(x_c))

    dx_c_dt = a * p_c ** n

    # print('dx_c_dt: ', str(dx_c_dt * 1000), ' mm/s')
    # print('p_c: ', str(p_c * 1e-6), ' MPa')

    # print(dx_c_dt)

    dV_c_dt = dx_c_dt * A_b(x_c)

    # print('Burn area: ', str(A_b(x_c)))

    if tools.is_choked(gamma_c, 1/M_c_recip, p_a, p_c, T_c, A_e, A_t):
        m_dot_out = tools.mass_flow_choked(A_t, p_c, T_c, gamma_c, 1/M_c_recip)
    else:
        m_dot_out = tools.mass_flow_subsonic_exit(gamma_c, 1/M_c_recip, p_a, p_c, T_c, A_e)
        # print('not choked')

    if m_dot_out < 0:
        m_dot_out = 0

    m_dot_prop = rho_p * dV_c_dt

    # print(str(m_dot_prop), ' - ', str(m_dot_out))

    dm_c_dt = m_dot_prop - m_dot_out

    dT_c_dt = ((T_prop - T_c) * c_p_prop * m_dot_prop) / (c_p_c * m_c)

    dc_p_c_dt = (c_p_prop - c_p_c) * m_dot_prop / m_c

    dM_c_recip_dt = (M_prop_recip - M_c_recip) * m_dot_prop / m_c

    dR_c_dt = R_univ * dM_c_recip_dt

    dgamma_c_dt = ((c_p_c - R_c) * dc_p_c_dt - c_p_c * (dc_p_c_dt - dR_c_dt)) / (c_p_c - R_c) ** 2

    dp_c_dt = (R_c * T_c / V_c) * dm_c_dt - (m_c * R_c * T_c / V_c ** 2) * dV_c_dt + (m_c * T_c / V_c) * dR_c_dt + (m_c * R_c / V_c) * dT_c_dt

    dydt = [dT_c_dt, dc_p_c_dt, dM_c_recip_dt, dm_c_dt, dp_c_dt, dV_c_dt, dx_c_dt, dR_c_dt, dgamma_c_dt]

    return dydt

y0 = [T_0, c_p_c_0, M_c_recip_0, m_c_0, p_c_0, V_c_0, x_c_0, R_c_0, gamma_c_0]

def main():

    t_end = 160
    time_step = 1e-3

    t = np.linspace(0, t_end, t_end/time_step + 1)

    sol = odeint(internal_ballistics, y0, t)
    # sol = odeint(internal_ballistics, [0, 160], y0)

    print(sol[-1,4])

    plt.plot(t, sol[:, 4])
    plt.show()

if __name__ == '__main__':
    main()
