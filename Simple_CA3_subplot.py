import random

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt, animation

from Predator_Prey_CA import Predator_Prey_CA, bc


class Simple_CA3(Predator_Prey_CA):
    """
    Implements the simple model from 'http://www.vuuren.co.za/Theses/Project2.pdf' except it's not broken
    """

    def __init__(self,
                 p_predator_death: float,
                 p_predator_birth: float,
                 p_prey_death: float,
                 p_prey_birth: float,
                 height: int,
                 width: int,
                 boundary_conditions: bc,
                 initial_ratio = None):
        super().__init__(3, height, width, boundary_conditions, initial_ratio=initial_ratio)
        self.p_predator_death = p_predator_death
        self.p_predator_birth = p_predator_birth
        self.p_prey_death = p_prey_death
        self.p_prey_birth = p_prey_birth

    def _apply_rules(self, position) -> int:
        """
        Here, 0 denotes empty, 1 denotes prey, 2 predator1, and 3 predator2.
        """
        y_pos, x_pos = position
        state = self.lattice[y_pos][x_pos]
        if state == 0:
            _, n_prey, n_predators, _ = self._moore_neighbourhood(position, 1)
            if n_prey == 0 or n_predators > 0:  # b a j s
                return 0
            else:
                p = random.uniform(0, 1)
                if p < (1 - self.p_prey_birth) ** n_prey:
                    return 1
                return 0
        elif state == 1:
            p = random.uniform(0, 1)
            _, _, n_predators, _ = self._moore_neighbourhood(position, 1)
            if p < (1 - self.p_prey_death) ** n_predators:
                return 1
            else:
                r = random.uniform(0, 1)
                if r < self.p_predator_birth:
                    return 2
                return 0
        elif state == 2:
            r = random.uniform(0, 1)
            if r < self.p_predator_birth:
                return 3

            p = random.uniform(0, 1)
            if p < self.p_predator_death:
                return 0
            else:
                return 2
        elif state == 3:
            p = random.uniform(0, 1)
            if p < self.p_predator_death:
                return 0
            else:
                return 3


def animate_ca():
    width = 50
    height = 50
    frames = 300
    ca = Simple_CA3(0.4, 0.9, 0.9, 0.2, height, width, bc.toroidal, initial_ratio=[0.7, 0.1, 0.1, 0.1])

    step_n_pred1 = []
    step_n_pred2 = []
    step_n_prey = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(ca.lattice, cmap=None, interpolation='nearest')
    fig.colorbar(im)

    def update(step):
        step_n_pred1.append((step, ca.count[2] / ca.tot_population))
        step_n_pred2.append((step, ca.count[3] / ca.tot_population))
        step_n_prey.append((step, ca.count[1] / ca.tot_population))
        ca.step()
        im.set_array(ca.lattice)
        return im,

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=5, blit=True, repeat=False)
    plt.show()

    plt.plot(*zip(*step_n_pred1), label='predator 1')
    plt.plot(*zip(*step_n_pred2), label='predator 2')
    plt.plot(*zip(*step_n_prey), label='prey')
    plt.ylabel('fraction of total population')
    plt.xlabel('Step')
    plt.legend()
    plt.show()


def statistics(values: list, epsilon: float, exclude: int = 2):
    converged = False
    n_required = 0
    mean = []
    standard_error = []
    X = list(range(exclude, len(values)))
    for n in range(exclude, len(values)):
        current_slice = values[:n]
        mean.append(np.mean(current_slice))
        standard_error.append(stats.sem(current_slice))

        if standard_error[-1] < epsilon and not converged and n > 2:
            n_required = n
            converged = True
    return X, mean, standard_error, n_required


def test():
    width = 50
    height = 50
    steps = 1000
    ca = Simple_CA3(0.4, 0.9, 0.9, 0.02, height, width, bc.toroidal, initial_ratio=[0.7, 0.1, 0.1, 0.1])

    epsilon = 0.001

    X = []
    n_prey = []
    n_predator1 = []
    n_predator2 = []

    for step in range(steps):
        X.append(step)
        n_prey.append(ca.count[1] / ca.tot_population)
        n_predator1.append(ca.count[2] / ca.tot_population)
        n_predator2.append(ca.count[3] / ca.tot_population)
        ca.step()

    exclude = 2
    _, mean_prey, sem_prey, n_required_prey = statistics(n_prey, epsilon, exclude=exclude)
    _, mean_predator1, sem_predator1, n_required_predator1 = statistics(n_predator1, epsilon, exclude=exclude)
    X_excluded, mean_predator2, sem_predator2, n_required_predator2 = statistics(n_predator2, epsilon, exclude=exclude)

    equilibrium_step = max(n_required_prey, n_required_predator1, n_required_predator2)
    intersection_prey = mean_prey[equilibrium_step - exclude]
    intersection_predator1 = mean_predator1[equilibrium_step - exclude]
    intersection_predator2 = mean_predator2[equilibrium_step - exclude]
    print('equilibrium ratios ({}% accuracy), required step: {}'.format(epsilon * 100, equilibrium_step))
    print('prey: {}, predator 1: {}, predator 2: {}'
          .format(intersection_prey, intersection_predator1, intersection_predator2))

    fig, ax = plt.subplots(2)
    ax[0].plot(X, n_prey, 'b', label='prey')
    ax[0].plot(X_excluded, mean_prey, 'b--', label='mean prey')
    ax[1].plot(X_excluded, sem_prey, 'b', label='SEM prey')

    ax[0].plot(X, n_predator1, 'r', label='predator 1')
    ax[0].plot(X_excluded, mean_predator1, 'r--', label='mean predator 1')
    ax[1].plot(X_excluded, sem_predator1, 'r', label='SEM predator 1')

    ax[0].plot(X, n_predator2, 'g', label='predator 2')
    ax[0].plot(X_excluded, mean_predator2, 'g--', label='mean predator 2')
    ax[1].plot(X_excluded, sem_predator2, 'g', label='SEM predator 2')

    ax[0].axvline(x=equilibrium_step, ls='--', color='k')
    ax[1].axvline(x=equilibrium_step, ls='--', color='k', label=r'SEM $\leq$ {}'.format(epsilon))

    ax[0].set_title('Fraction of total population')
    ax[1].set_title('SEM (Standard error of the mean)')
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[0].set(xlabel='$n$ steps')
    ax[0].set(xlim=[-2, steps])
    ax[0].set(ylabel='ratio')
    ax[1].set(xlabel='$n$ steps')
    ax[1].set(xlim=[-2, steps])
    ax[1].set(ylabel=r'$\sigma/\sqrt{N}$')
    ax.flat[0].label_outer()
    plt.show()


if __name__ == '__main__':
    animate_ca()
    #test()
