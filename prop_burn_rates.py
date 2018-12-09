prop_boundaries = [0, 39, 87, 317, 355, 380]
oxamide_conc = [0.014, 0.086, 0.15, 0.086, 0.006]
flame_temp = [ 2172, 1889, 1673, 1889, 2172]
spec_heat = [1961, 1881, 1860, 1881, 1961]
mol_mass_recip = [1/0.02234, 1/0.02187, 1/0.02154, 1/0.02187, 1/0.02234]

def burn_rate_model(w_om):
    """Burn rate vs oxamide content model."""
    n = 0.45    # Burn rate exponent [units: dimensionless]
    # un-doped burn rate coef [units: pascal**(-n) meter second**-1].
    a_0 = 7 * (1e6)**(-n) * 1e-3
    lamb = 7.0    # Oxamide parameter [units: dimensionless].
    return a_0 * (1 - w_om) / (1 + lamb * w_om)


def prop_values(x_c):

    x_test = x_c * 1000

    a_test = 0

    for index in range(len(prop_boundaries) - 1):
        if prop_boundaries[index] <= x_test  and x_test < prop_boundaries[index + 1]:
            oxamide_conc_test = oxamide_conc[index]
            a_test = burn_rate_model(oxamide_conc_test) # burn rate coefficient in m/s
            temp_test = flame_temp[index]
            spec_heat_test = spec_heat[index]
            mol_mass_recip_test = mol_mass_recip[index]
            break

    return [a_test, temp_test, spec_heat_test, mol_mass_recip_test]

def main():

    test1 = burn_rate_coeff(30/1000)
    print(test1)
    test2 = burn_rate_coeff(38/1000)
    print(test2)
    test3 = burn_rate_coeff(100/1000)
    print(test3)
    test4 = burn_rate_coeff(330/1000)
    print(test4)

if __name__ == '__main__':
    main()