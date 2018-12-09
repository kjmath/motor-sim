import numpy as np 

x = [0, 2.5, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 380]
area = [579, 602, 520, 468, 552, 892, 1342, 1585, 1664, 1692, 1458, 884, 321]

def A_b(x_c):
    """Find burn area given burn distance.

    Inputs:
        x_c: burn distance along grain [units: m]

    Returns:
        A_b: burn area [units: m]"""

    x_test = x_c * 1000.
    area_test = np.interp(x_test, x, area)

    return area_test * 1e-6


def main():
    print(A_b(0.038))

    print('Kn: ', str(A_b(0.038)/((3.22/2/1000)**2 * 3.14159)))

if __name__ == '__main__':
    main()
