import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad

LOWER_BOUND = 0
UPPER_BOUND = 1
# Known density function
RHO_FUNCTION = lambda r: 1 + r


def find_integral(function, lower_bound, upper_bound):
    'Calculates integral of rho(r) to find mu1 and mu2'
    return quad(function, lower_bound, upper_bound)[0]

def find_optimal_weights(s, integral_r,  integral_rcubed):
    integral_1 = find_integral(lambda r: r**2 * (r-s)**2, 0, 1)
    integral_2 = find_integral(lambda r: r**4 * (r-s)**2, 0, 1)
    integral_3 = find_integral(lambda r: r**6 * (r-s)**2, 0, 1)
    
    A = np.array([[integral_1, integral_2, -integral_r],
                [integral_2, integral_3,-integral_rcubed],
                [integral_r, integral_rcubed, 0]])
    b = np.array([0, 0, 1])
    x = np.linalg.solve(A, b)
    return x

def backus_gilbert_calc(rho_function, integral_r, integral_rcubed, 
                        lower_bound, upper_bound):
    'Computes the Backus-Gilbert method'
    # mu1 (mass-related moment) tells us about amount of material in the disc
    # mu2 (inertia_related moment) tells us how the material is distributed relative to the centre
    mu1 = find_integral(lambda r: rho_function(r) * r, lower_bound, upper_bound)
    mu2 = find_integral(lambda r: rho_function(r) * r**3, lower_bound, upper_bound)
    s_positions  = np.linspace(0.001, 0.999, 100)
    rho_prediction_values = []
    for s in s_positions:
        x = find_optimal_weights(s, integral_r,  integral_rcubed)
        c1 = x[0]
        c2 = x[1]
        rho_prediction = (c1 * mu1) + (c2 * mu2)
        rho_prediction_values.append(rho_prediction)
    return s_positions, rho_prediction_values

def plot_rho_predicted(s_positions, rho_prediction_values):
    plt.plot(s_positions, rho_prediction_values, label = 'Predicted density')
    plt.xlabel('Radius (s)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def main():
    integral_r = find_integral(lambda r: r, LOWER_BOUND, UPPER_BOUND)
    integral_rcubed = find_integral(lambda r: r**3, LOWER_BOUND, UPPER_BOUND)
    s_positions, rho_prediction_values = backus_gilbert_calc(RHO_FUNCTION, integral_r, 
                                                             integral_rcubed, LOWER_BOUND, UPPER_BOUND)
    plot_rho_predicted(s_positions, rho_prediction_values)

if __name__ == '__main__':
    main()
