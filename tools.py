from scipy.optimize import fsolve
import math

R_univ = 8.314

def mach_from_pr(p_c, p_e, gamma):
    ''' Find the exit Mach number from the pressure ratio.

    Arugments:
        p_c: Nozzle stagnation chamber pressure [units: pascal].
        p_e: Nozzle exit static pressure [units: pascal].
        gamma: Exhaust gas ratio of specific heats [units: none].

    Returns:
        scalar: Exit Mach number [units: none].
    '''
    return (2 / (gamma - 1) * ((p_e / p_c)**((1 - gamma) / gamma) -1))**0.5

def mass_flow_subsonic_exit(gamma, m_molar, p_a, p_c, T_c, A_e):
    """Find the mass flow through a nozzle with a subsonic exit velocity.

    Reference: Mechanics and Thermodynamics of Propulsion, 2nd Edition, Equations 3.6, 3.13

    Arguments:
        gamma (scalar): Exhaust gas ratio of specific heats [units: dimensionless].
        m_molar (scalar): Exhaust gas mean molar mass [units: kilogram mole**-1].
        p_a (scalar): Atmospheric pressure [units: pascal].
        p_c (scalar): Nozzle stagnation pressure [units: pascal].
        T_c (scalar): Nozzle stagnation temperature [units: kelvin].
        A_e (scalar): Nozzle exit area [units: meter**2].

    Returns:
        scalar: Mass flow through the nozzle [units: kg second**-1].
        """
    R = R_univ/m_molar
    if p_c < p_a:
        p_c = p_a
    M = mach_from_pr(p_c, p_a, gamma) # Mach number at exit
    return A_e*p_c*(gamma/(R*T_c))**(0.5)*M*(1.+(gamma-1.)/2.*M**2.)**(-(gamma+1.)/(2.*(gamma-1.)))

def mass_flow_choked(A_t, p_c, T_c, gamma, m_molar):

    return A_t * p_c / math.sqrt(T_c) * math.sqrt(gamma / (R_univ/m_molar)) * ((gamma + 1) / 2) ** ((1 - gamma)/(2 * gamma - 2))

def is_choked(gamma, m_molar, p_a, p_c, T_c, A_e, A_t):
    """Determine whether nozzle flow is choked.

    Reference: Mechanics and Thermodynamics of Propulsion, 2nd Edition, Equations 3.13 and 3.14

    Arguments:
        gamma (scalar): Exhaust gas ratio of specific heats [units: dimensionless].
        m_molar (scalar): Exhaust gas mean molar mass [units: kilogram mole**-1].
        p_a (scalar): Atmospheric pressure [units: pascal].
        p_c (scalar): Nozzle stagnation pressure [units: pascal].
        T_c (scalar): Nozzle stagnation temperature [units: Kelvin].
        A_e (scalar): Nozzle exit area [units: meter**2].
        A_t (scalar): Nozzle throat area [units: meter**2].

    Returns:
        bool: True if flow is choked, false otherwise.
    """

    def equate_mass_flow(p_find):
        """Equate mass flows using subsonic and supersonic models."""
        mass_flow_sub = mass_flow_subsonic_exit(gamma, m_molar, p_a, p_find, T_c, A_e)
        mass_flow_sup = mass_flow_choked(A_t, p_find, T_c, gamma, m_molar)
        return mass_flow_sub - mass_flow_sup

    p_critical = fsolve(equate_mass_flow, 5*p_a)

    if p_c < p_a:
        p_c = p_a
    return p_c >= p_critical[0]

def p_critical(gamma, m_molar, p_a, p_c, T_c, A_e, A_t):

    def equate_mass_flow(p_find):
        """Equate mass flows using subsonic and supersonic models."""
        mass_flow_sub = mass_flow_subsonic_exit(gamma, m_molar, p_a, p_find, T_c, A_e)
        mass_flow_sup = mass_flow_choked(A_t, p_find, T_c, gamma, m_molar)
        return mass_flow_sub - mass_flow_sup

    p_critical = fsolve(equate_mass_flow, 5*p_a)

    return p_critical