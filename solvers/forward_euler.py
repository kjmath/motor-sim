"""Foward Euler integration scheme."""
import numpy as np

def forward_euler(func, y0, t):

    y = np.full((len(t), len(y0)), None)

    y[0, :] = y0

    for index in range(1, len(t)):
        delta_t = t[index] - t[index - 1]
        y[index, :] = y[index - 1, :] + np.array(func(y[index - 1, :], t[index - 1])) * delta_t

    return y

def main():
    
    def linear(y, t):
        x1, x2 = y
        dydt = [2, 3]

        return dydt

    y0 = [0, 0]
    t = np.linspace(0, 5, 51)

    sol = forward_euler(linear, y0, t)

if __name__ == '__main__':
    main()
