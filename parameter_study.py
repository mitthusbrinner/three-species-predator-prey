from enum import Enum

import numpy as np
from matplotlib import pyplot as plt

from Predator_Prey_CA import bc
from Simple_CA3 import Simple_CA3
from scipy import stats


class errorType(Enum):
    Cv = 0
    SEM = 1


def parameter_study(T: int,
                    probabilities: list,
                    values_to_test: list,
                    boundary_conditions: bc,
                    initial_ratio: list,
                    width: int = 50,
                    height: int = 50,
                    ):
    """
    Example:
        We want to vary p_predator1_death, then we let probabilities = ['x', 0.9, 0.4, 0.4, 0.3, 0.8]
    """
    idx_probability_to_test = probabilities.index('x')
    equilibrium_ratios = []
    for value in values_to_test:
        probabilities[idx_probability_to_test] = value
        ca = Simple_CA3(
            probabilities[0], probabilities[1], probabilities[2], probabilities[3], probabilities[4], probabilities[5],
            height, width, boundary_conditions, initial_ratio=initial_ratio)

        for _ in range(T):
            ca.step()

        total_population = ca.tot_population
        equilibrium_ratios.append((
            ca.count[1] / total_population,
            ca.count[2] / total_population,
            ca.count[3] / total_population
        ))
    return equilibrium_ratios


def main():
    values_to_test = [k / 10 for k in range(10 + 1)]
    equilibrium_ratios = parameter_study(errorType.SEM,
                                         0.005,
                                         [0.9, 0.8, 0.2, 'x', 0.05, 0.9],
                                         values_to_test,
                                         bc.toroidal,
                                         [0.7, 0.1, 0.1, 0.1],
                                         exclude=100
                                         )
    plt.plot(values_to_test, equilibrium_ratios)
    plt.ylabel(r'Mean ratio at equilibrium')
    plt.xlabel(r'Probability')
    plt.legend(['Prey', 'Predator 1', 'Predator 2'])
    plt.show()


if __name__ == '__main__':
    main()
