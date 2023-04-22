import random

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt, animation
from tqdm import tqdm

from Predator_Prey_CA import Predator_Prey_CA, bc


class Simple_CA3(Predator_Prey_CA):
    """
    Implements the simple model from 'http://www.vuuren.co.za/Theses/Project2.pdf' except it's not broken
    """

    def __init__(self,
                 p_prey_death: float,
                 p_prey_birth: float,
                 p_predator1_death: float,
                 p_predator1_birth: float,
                 p_predator2_death: float,
                 p_predator2_birth: float,
                 height: int,
                 width: int,
                 boundary_conditions: bc,
                 initial_ratio=None):
        super().__init__(3, height, width, boundary_conditions, initial_ratio=initial_ratio)
        self.p_prey_death = p_prey_death
        self.p_prey_birth = p_prey_birth
        self.p_predator1_death = p_predator1_death
        self.p_predator1_birth = p_predator1_birth
        self.p_predator2_death = p_predator2_death
        self.p_predator2_birth = p_predator2_birth

    def _apply_rules(self, position) -> int:
        """
        Here, 0 denotes empty, 1 denotes prey, 2 predator1, and 3 predator2.
        """
        y_pos, x_pos = position
        state = self.lattice[y_pos][x_pos]
        if state == 0:
            _, n_prey, n_predators, _ = self._moore_neighbourhood(position, 1)
            if n_prey == 0:
                return 0
            else:
                p = random.uniform(0, 1)
                if p < (1 - self.p_prey_birth) ** n_prey:  # crowding
                    return 1
                return 0
        elif state == 1:
            p = random.uniform(0, 1)
            _, _, n_predators, _ = self._moore_neighbourhood(position, 1)
            if p < (1 - self.p_prey_death) ** n_predators:  # not hunted by predator 1
                return 1
            else:  # hunted by predator 1
                r = random.uniform(0, 1)
                if r < self.p_predator1_birth:  # prey dies and become predator 1
                    return 2
                return 0
        elif state == 2:
            p = random.uniform(0, 1)
            _, _, _, n_predators = self._moore_neighbourhood(position, 1)
            if p < (1 - self.p_predator1_death) ** n_predators:  # not hunted by predator 2
                p = random.uniform(0, 1)
                if p < self.p_predator1_death:  # natural death
                    return 0
                else:
                    return 2  # survive
            else:  # hunted by predator 2
                r = random.uniform(0, 1)
                if r < self.p_predator2_birth:  # prey dies and become predator 1
                    return 3
                return 0

        elif state == 3:
            p = random.uniform(0, 1)
            if p < self.p_predator2_death:
                return 0
            else:
                return 3


def animate_ca():
    width = 50
    height = 50
    frames = 350
    ca = Simple_CA3(0.9, 0.8, 0.2, 1, 0.05, 0.9, height, width, bc.toroidal, initial_ratio=[0.7, 0.1, 0.1, 0.1])

    step_n_pred1 = []
    step_n_pred2 = []
    step_n_prey = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(ca.lattice, cmap=None, interpolation='nearest')
    fig.colorbar(im)

    def update(step):
        total_population = ca.tot_population
        if total_population != 0:
            step_n_pred1.append((step, ca.count[2] / ca.tot_population))
            step_n_pred2.append((step, ca.count[3] / ca.tot_population))
            step_n_prey.append((step, ca.count[1] / ca.tot_population))
        else:
            step_n_pred1.append((step, 0))
            step_n_pred2.append((step, 0))
            step_n_prey.append((step, 0))
        ca.step()
        im.set_array(ca.lattice)
        return im,

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=10, blit=True, repeat=False)
    plt.show()

    plt.plot(*zip(*step_n_prey), label='prey')
    plt.plot(*zip(*step_n_pred1), label='predator 1')
    plt.plot(*zip(*step_n_pred2), label='predator 2')

    plt.ylabel('fraction of total population')
    plt.xlabel('Step')
    plt.legend()
    plt.show()


def statistics(values: list, epsilon: float, exclude: int = 2):
    converged = False
    n_required = 0
    mean = []
    standard_error = []
    cv = []
    X = list(range(exclude, len(values)))
    for n in range(exclude, len(values)):
        current_slice = values[:n]
        mean.append(np.mean(current_slice))
        standard_error.append(stats.sem(current_slice))
        cv.append(stats.variation(current_slice))

        if standard_error[-1] < epsilon and not converged and n > exclude:
            n_required = n
            converged = True
    return X, mean, standard_error, cv, n_required


def test():
    width = 50
    height = 50
    steps = 200
    ca = Simple_CA3(0.9, 0.8, 0.2, 0.9, 0.05, 0.9, height, width, bc.toroidal, initial_ratio=[0.7, 0.1, 0.1, 0.1])

    epsilon = 0.005

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

    exclude = 100
    _, mean_prey, sem_prey, cv_prey, n_required_prey = statistics(n_prey, epsilon, exclude=exclude)
    _, mean_predator1, sem_predator1, cv_predator1, n_required_predator1 = statistics(n_predator1, epsilon,
                                                                                      exclude=exclude)
    X_excluded, mean_predator2, sem_predator2, cv_predator2, n_required_predator2 = statistics(n_predator2, epsilon,
                                                                                               exclude=exclude)

    equilibrium_step = max(n_required_prey, n_required_predator1, n_required_predator2)
    intersection_prey = mean_prey[equilibrium_step - exclude]
    intersection_predator1 = mean_predator1[equilibrium_step - exclude]
    intersection_predator2 = mean_predator2[equilibrium_step - exclude]
    print('equilibrium ratios ({}% accuracy)'.format(epsilon * 100))
    print('prey: {}, predator 1: {}, predator 2: {}'
          .format(intersection_prey, intersection_predator1, intersection_predator2))

    M_desired = 100
    M = M_desired if steps - equilibrium_step > M_desired else steps - equilibrium_step
    X_converged_mean = list(range(equilibrium_step, equilibrium_step + M))
    converged_mean_prey = np.mean(n_prey[equilibrium_step:equilibrium_step + M])
    converged_mean_predator1 = np.mean(n_predator1[equilibrium_step:equilibrium_step + M])
    converged_mean_predator2 = np.mean(n_predator2[equilibrium_step:equilibrium_step + M])

    plt.plot(X, n_prey, 'b', label='prey ratio')
    plt.plot(X_excluded, mean_prey, 'b--', label='mean prey')
    plt.plot(X_excluded, sem_prey, 'tab:blue', label='sem prey')
    plt.plot(X_excluded, cv_prey, 'tab:blue', label='cv prey')
    # plt.plot(X_converged_mean, [converged_mean_prey] * M, 'b-.')

    plt.plot(X, n_predator1, 'r', label='ratio predator 1')
    plt.plot(X_excluded, mean_predator1, 'r--', label='mean predator 1')
    plt.plot(X_excluded, sem_predator1, 'tab:red', label='sem predator 1')
    plt.plot(X_excluded, cv_predator1, 'tab:red', label='cv predator 1')
    # plt.plot(X_converged_mean, [converged_mean_predator1] * M, 'r-.', label='sem prey')

    plt.plot(X, n_predator2, 'g', label='ratio predator 2')
    plt.plot(X_excluded, mean_predator2, 'g--', label='mean predator 2')
    plt.plot(X_excluded, sem_predator2, 'tab:green', label='sem predator 2')
    plt.plot(X_excluded, cv_predator2, 'tab:green', label='cv predator 1')
    # plt.plot(X_converged_mean, [converged_mean_predator2] * M, 'g-.')

    plt.axvline(x=equilibrium_step, ls='--', color='k')
    plt.xlabel('$n$ steps')
    # plt.legend()
    plt.show()


def error_plot():
    width = 50
    height = 50
    steps = 1000

    n_simulations = 60
    epsilon = 0.01

    prey_sem = []
    predator1_sem = []
    predator2_sem = []

    prey_cv = []
    predator1_cv = []
    predator2_cv = []

    prey_mean_ratios = []
    predator1_mean_ratios = []
    predator2_mean_ratios = []
    for simulation in tqdm(range(n_simulations + 1)):
        ca = Simple_CA3(0.9925629160207855, 0.154667180093502, 0.39186958820898643, 0.7644954722324826,
                        0.10463787447507766,
                        0.27740597266372047, height, width, bc.toroidal, initial_ratio=[0.901, 0.033, 0.033, 0.033])
        # print('{0:.1f}%'.format(simulation / n_simulations * 100))  # print progress
        prey_ratio = []
        predator1_ratio = []
        predator2_ratio = []
        for step in range(steps):
            prey_ratio.append(ca.count[1] / ca.tot_population)
            predator1_ratio.append(ca.count[2] / ca.tot_population)
            predator2_ratio.append(ca.count[3] / ca.tot_population)
            ca.step()
        prey_mean_ratios.append(np.mean(prey_ratio))
        predator1_mean_ratios.append(np.mean(predator1_ratio))
        predator2_mean_ratios.append(np.mean(predator2_ratio))

        if simulation > 0:
            prey_sem.append(stats.sem(prey_mean_ratios))
            predator1_sem.append(stats.sem(predator1_mean_ratios))
            predator2_sem.append(stats.sem(predator2_mean_ratios))

            prey_cv.append(stats.variation(prey_mean_ratios))
            predator1_cv.append(stats.variation(predator1_mean_ratios))
            predator2_cv.append(stats.variation(predator2_mean_ratios))

    X = list(range(1, n_simulations + 1))
    plt.title(r'{} steps'.format(steps))

    '''
    plt.plot(X, prey_sem)
    plt.plot(X, predator1_sem)
    plt.plot(X, predator2_sem)
    plt.ylabel(r'$\sigma / \sqrt{n}$')
    '''

    plt.plot(X, prey_sem)
    plt.plot(X, predator1_sem)
    plt.plot(X, predator2_sem)
    plt.ylabel(r'$C_v$')
    plt.title('$C_v$ of ratio of total population, $T={}$'.format(steps))

    plt.axhline(y=epsilon, color='k', linestyle='--')
    plt.legend(['Prey', 'Predator 1', 'Predator 1'])
    plt.xlabel(r'$n$ simulations')

    plt.show()


def error_plot_v2():
    width = 50
    height = 50
    n_simulations = 100
    T_values = [10000, 1000, 100, 10]
    all_max_cv_values = []
    all_max_sem_values = []
    with tqdm(total=sum(T_values) * n_simulations) as pbar:
        for T in T_values:
            max_cv_values = []
            max_sem_values = []

            prey_mean_ratios = []
            predator1_mean_ratios = []
            predator2_mean_ratios = []
            for simulation in range(n_simulations):
                ca = Simple_CA3(0.9925629160207855, 0.154667180093502, 0.39186958820898643, 0.7644954722324826,
                                0.10463787447507766,
                                0.27740597266372047, height, width, bc.toroidal,
                                initial_ratio=[0.901, 0.033, 0.033, 0.033])
                prey_ratio = []
                predator1_ratio = []
                predator2_ratio = []
                for step in range(T):
                    prey_ratio.append(ca.count[1] / ca.tot_population)
                    predator1_ratio.append(ca.count[2] / ca.tot_population)
                    predator2_ratio.append(ca.count[3] / ca.tot_population)
                    ca.step()
                    pbar.update(1)
                prey_mean_ratios.append(np.mean(prey_ratio))
                predator1_mean_ratios.append(np.mean(predator1_ratio))
                predator2_mean_ratios.append(np.mean(predator2_ratio))

                if simulation > 0:
                    max_cv_values.append(
                        max(
                            stats.variation(prey_mean_ratios),
                            stats.variation(predator1_mean_ratios),
                            stats.variation(predator2_mean_ratios)
                        ))
                    max_sem_values.append(
                        max(
                            stats.sem(prey_mean_ratios),
                            stats.sem(predator1_mean_ratios),
                            stats.sem(predator2_mean_ratios)
                        ))
                pbar.update(1)
            all_max_cv_values.append(max_cv_values)
            all_max_sem_values.append(max_sem_values)

    X = list(range(1, n_simulations))
    for i, errors in enumerate(all_max_cv_values):
        plt.plot(X, errors, label='$T={}$'.format(T_values[i]))

    plt.yscale('log')
    plt.xlabel('$n$ simulations')
    plt.ylabel('max $C_v$')
    plt.title('$C_v$ of ratio of total population for varying $T$')
    plt.legend()
    plt.show()

    for i, errors in enumerate(all_max_sem_values):
        plt.plot(X, errors, label='$T={}$'.format(T_values[i]))

    plt.yscale('log')
    plt.xlabel('$n$ simulations')
    plt.ylabel(r'max $\sigma / \sqrt{n}$')
    plt.title('Standard error of ratio of total population for varying $T$')
    plt.legend()
    plt.show()


def lattice_dimension():
    grid_sizes = [10, 20, 30, 40, 50, 60, 70, 80]
    T = 1000
    M = 10

    z_values = []
    err = []
    with tqdm(total=len(grid_sizes) * T * M) as pbar:
        for size in grid_sizes:
            current_z_values = []
            for _ in range(M):
                ca = Simple_CA3(0.9925629160207855, 0.154667180093502, 0.39186958820898643, 0.7644954722324826,
                                0.10463787447507766,
                                0.27740597266372047, size, size, bc.toroidal,
                                initial_ratio=[0.901, 0.033, 0.033, 0.033])
                for step in range(T):
                    ca.step()
                    if step % 10 == 0:
                        pbar.update(10)

                total_population = ca.tot_population
                if total_population > 0:
                    alpha = ca.count[1] / total_population  # prey ratio
                    beta = ca.count[2] / total_population  # predator 1 ratio
                    gamma = ca.count[3] / total_population  # predator 2 ratio
                    z = alpha * beta * gamma * 27
                current_z_values.append(z)
            z_values.append(np.mean(current_z_values))
            err.append(np.std(current_z_values))

    plt.errorbar(grid_sizes, z_values, yerr=err, fmt="o")
    plt.plot(grid_sizes, z_values, color='tab:blue')
    plt.xlabel(r'Lattice width and height')
    plt.ylabel(r'Normalized $z$-value')
    plt.title('Normalized $z$-value as a function of lattice size')
    plt.show()


def boundary_conditioon():
    boundary_conditions = [bc.toroidal, bc.reflective]
    width = 50
    T = 1000
    M = 10

    z_values = []
    err = []
    with tqdm(total=len(boundary_conditions) * T * M) as pbar:
        for bound_cond in boundary_conditions:
            current_z_values = []
            for _ in range(M):
                ca = Simple_CA3(0.9925629160207855, 0.154667180093502, 0.39186958820898643, 0.7644954722324826,
                                0.10463787447507766,
                                0.27740597266372047, width, width, bound_cond,
                                initial_ratio=[0.901, 0.033, 0.033, 0.033])
                for step in range(T):
                    ca.step()
                    if step % 10 == 0:
                        pbar.update(10)

                total_population = ca.tot_population
                if total_population > 0:
                    alpha = ca.count[1] / total_population  # prey ratio
                    beta = ca.count[2] / total_population  # predator 1 ratio
                    gamma = ca.count[3] / total_population  # predator 2 ratio
                    z = alpha * beta * gamma * 27
                current_z_values.append(z)
            z_values.append(np.mean(current_z_values))
            err.append(np.std(current_z_values))

    print('Toroidal: \n mu = {}, sigma = {}'.format(z_values[0], err[0]))
    print('Reflective: \n mu = {}, sigma = {}'.format(z_values[1], err[1]))


def sem_plot_bc():
    width = 50
    height = 50
    increment = 2
    n_simulations = list(range(0, 80 + increment, increment))
    T = 1000
    boundary_conditions = [bc.toroidal, bc.reflective]
    all_max_cv_values = []
    all_max_sem_values = []
    with tqdm(total=len(boundary_conditions) * len(n_simulations) * T) as pbar:
        for bound_cond in boundary_conditions:
            max_cv_values = []
            max_sem_values = []

            prey_mean_ratios = []
            predator1_mean_ratios = []
            predator2_mean_ratios = []
            for simulation in n_simulations:
                ca = Simple_CA3(0.9925629160207855, 0.154667180093502, 0.39186958820898643, 0.7644954722324826,
                                0.10463787447507766,
                                0.27740597266372047, height, width, bound_cond,
                                initial_ratio=[0.901, 0.033, 0.033, 0.033])
                prey_ratio = []
                predator1_ratio = []
                predator2_ratio = []
                for step in range(T):
                    prey_ratio.append(ca.count[1] / ca.tot_population)
                    predator1_ratio.append(ca.count[2] / ca.tot_population)
                    predator2_ratio.append(ca.count[3] / ca.tot_population)
                    ca.step()
                    if step % 10 == 0:
                        pbar.update(10)
                prey_mean_ratios.append(np.mean(prey_ratio))
                predator1_mean_ratios.append(np.mean(predator1_ratio))
                predator2_mean_ratios.append(np.mean(predator2_ratio))

                if simulation > 0:
                    max_cv_values.append(
                        max(
                            stats.variation(prey_mean_ratios),
                            stats.variation(predator1_mean_ratios),
                            stats.variation(predator2_mean_ratios)
                        ))
                    max_sem_values.append(
                        max(
                            stats.sem(prey_mean_ratios),
                            stats.sem(predator1_mean_ratios),
                            stats.sem(predator2_mean_ratios)
                        ))
            all_max_cv_values.append(max_cv_values)
            all_max_sem_values.append(max_sem_values)

    legend_strings = ['Toroidal', 'Reflective']
    X = n_simulations[1:]
    for i, errors in enumerate(all_max_cv_values):
        plt.plot(X, errors, label=legend_strings[i])

    plt.yscale('log')
    plt.xlabel('$n$ simulations')
    plt.ylabel('max $C_v$')
    plt.title('$C_v$ of ratio of total population for different boundary conditions')
    plt.legend()
    plt.show()

    for i, errors in enumerate(all_max_sem_values):
        plt.plot(X, errors, label=legend_strings[i])

    plt.yscale('log')
    plt.xlabel('$n$ simulations')
    plt.ylabel(r'max $\sigma / \sqrt{n}$')
    plt.title('Standard error of ratio of total population for different boundary conditions')
    plt.legend()
    plt.show()


def initial_proportion():
    number_of_random_initial_ratios = 100
    width = 50
    T = 1000

    z_values = []
    with tqdm(total=number_of_random_initial_ratios * T) as pbar:
        for _ in range(number_of_random_initial_ratios):
            ca = Simple_CA3(0.9925629160207855, 0.154667180093502, 0.39186958820898643, 0.7644954722324826,
                            0.10463787447507766,
                            0.27740597266372047, width, width, bc.toroidal)
            for step in range(T):
                ca.step()
                if step % 10 == 0:
                    pbar.update(10)

            total_population = ca.tot_population
            if total_population > 0:
                alpha = ca.count[1] / total_population  # prey ratio
                beta = ca.count[2] / total_population  # predator 1 ratio
                gamma = ca.count[3] / total_population  # predator 2 ratio
                z = alpha * beta * gamma * 27
            z_values.append(z)

    plt.hist(z_values, density=False, bins=30)
    plt.ylabel('Frequency')
    plt.xlabel('normalized $z$-value')
    plt.title('normalized $z$-value for {} randomized initial conditions'.format(number_of_random_initial_ratios))
    plt.show()


def lattice_size_error():
    n_simulations = 100
    lattice_sizes = [10, 20, 50, 60, 80]
    T = 1000
    all_max_cv_values = []
    all_max_sem_values = []
    with tqdm(total=len(lattice_sizes) * n_simulations * T) as pbar:
        for lattice_size in lattice_sizes:
            max_cv_values = []
            max_sem_values = []

            prey_mean_ratios = []
            predator1_mean_ratios = []
            predator2_mean_ratios = []
            for simulation in range(n_simulations):
                ca = Simple_CA3(0.9925629160207855, 0.154667180093502, 0.39186958820898643, 0.7644954722324826,
                                0.10463787447507766,
                                0.27740597266372047, lattice_size, lattice_size, bc.toroidal,
                                initial_ratio=[0.901, 0.033, 0.033, 0.033])
                prey_ratio = []
                predator1_ratio = []
                predator2_ratio = []
                for step in range(T):
                    prey_ratio.append(ca.count[1] / ca.tot_population)
                    predator1_ratio.append(ca.count[2] / ca.tot_population)
                    predator2_ratio.append(ca.count[3] / ca.tot_population)
                    ca.step()
                    if step % 10 == 0:
                        pbar.update(10)
                prey_mean_ratios.append(np.mean(prey_ratio))
                predator1_mean_ratios.append(np.mean(predator1_ratio))
                predator2_mean_ratios.append(np.mean(predator2_ratio))

                if simulation > 0:
                    max_cv_values.append(
                        max(
                            stats.variation(prey_mean_ratios),
                            stats.variation(predator1_mean_ratios),
                            stats.variation(predator2_mean_ratios)
                        ))
                    max_sem_values.append(
                        max(
                            stats.sem(prey_mean_ratios),
                            stats.sem(predator1_mean_ratios),
                            stats.sem(predator2_mean_ratios)
                        ))
            all_max_cv_values.append(max_cv_values)
            all_max_sem_values.append(max_sem_values)

    X = list(range(1, n_simulations))
    for i, errors in enumerate(all_max_cv_values):
        plt.plot(X, errors, label='$s={}$'.format(lattice_sizes[i]))

    plt.yscale('log')
    plt.xlabel('$n$ simulations')
    plt.ylabel('max $C_v$')
    plt.title('$C_v$ of ratio of total population for varying grid sizes $s$')
    plt.legend()
    plt.show()

    for i, errors in enumerate(all_max_sem_values):
        plt.plot(X, errors, label='$s={}$'.format(lattice_sizes[i]))

    plt.yscale('log')
    plt.xlabel('$n$ simulations')
    plt.ylabel(r'max $\sigma / \sqrt{n}$')
    plt.title('Standard error of ratio of total population for varying grid sizes $s$')
    plt.legend()
    plt.show()

    def inital_ratio__error():
        n_simulations = 100
        lattice_sizes = [10, 20, 50, 60, 80]
        T = 1000
        all_max_cv_values = []
        all_max_sem_values = []
        with tqdm(total=len(lattice_sizes) * n_simulations * T) as pbar:
            for lattice_size in lattice_sizes:
                max_cv_values = []
                max_sem_values = []

                prey_mean_ratios = []
                predator1_mean_ratios = []
                predator2_mean_ratios = []
                for simulation in range(n_simulations):
                    ca = Simple_CA3(0.9925629160207855, 0.154667180093502, 0.39186958820898643, 0.7644954722324826,
                                    0.10463787447507766,
                                    0.27740597266372047, 50, 50, bc.toroidal,
                                    initial_ratio=[0.901, 0.033, 0.033, 0.033])
                    prey_ratio = []
                    predator1_ratio = []
                    predator2_ratio = []
                    for step in range(T):
                        prey_ratio.append(ca.count[1] / ca.tot_population)
                        predator1_ratio.append(ca.count[2] / ca.tot_population)
                        predator2_ratio.append(ca.count[3] / ca.tot_population)
                        ca.step()
                        if step % 10 == 0:
                            pbar.update(10)
                    prey_mean_ratios.append(np.mean(prey_ratio))
                    predator1_mean_ratios.append(np.mean(predator1_ratio))
                    predator2_mean_ratios.append(np.mean(predator2_ratio))

                    if simulation > 0:
                        max_cv_values.append(
                            max(
                                stats.variation(prey_mean_ratios),
                                stats.variation(predator1_mean_ratios),
                                stats.variation(predator2_mean_ratios)
                            ))
                        max_sem_values.append(
                            max(
                                stats.sem(prey_mean_ratios),
                                stats.sem(predator1_mean_ratios),
                                stats.sem(predator2_mean_ratios)
                            ))
                all_max_cv_values.append(max_cv_values)
                all_max_sem_values.append(max_sem_values)

        X = list(range(1, n_simulations))
        for i, errors in enumerate(all_max_cv_values):
            plt.plot(X, errors, label='$s={}$'.format(lattice_sizes[i]))

        plt.yscale('log')
        plt.xlabel('$n$ simulations')
        plt.ylabel('max $C_v$')
        plt.title('$C_v$ of ratio of total population for varying grid sizes $s$')
        plt.legend()
        plt.show()

        for i, errors in enumerate(all_max_sem_values):
            plt.plot(X, errors, label='$s={}$'.format(lattice_sizes[i]))

        plt.yscale('log')
        plt.xlabel('$n$ simulations')
        plt.ylabel(r'max $\sigma / \sqrt{n}$')
        plt.title('Standard error of ratio of total population for varying grid sizes $s$')
        plt.legend()
        plt.show()


def initial_ratio__error():
    n_simulations = 100
    n_random_initial = 10
    T = 1000
    all_max_cv_values = []
    all_max_sem_values = []
    with tqdm(total=n_random_initial * n_simulations * T) as pbar:
        for _ in range(n_random_initial):
            max_cv_values = []
            max_sem_values = []

            prey_mean_ratios = []
            predator1_mean_ratios = []
            predator2_mean_ratios = []
            for simulation in range(n_simulations):
                ca = Simple_CA3(0.9925629160207855, 0.154667180093502, 0.39186958820898643, 0.7644954722324826,
                                0.10463787447507766,
                                0.27740597266372047, 50, 50, bc.toroidal,
                                initial_ratio=[0.901, 0.033, 0.033, 0.033])
                prey_ratio = []
                predator1_ratio = []
                predator2_ratio = []
                for step in range(T):
                    prey_ratio.append(ca.count[1] / ca.tot_population)
                    predator1_ratio.append(ca.count[2] / ca.tot_population)
                    predator2_ratio.append(ca.count[3] / ca.tot_population)
                    ca.step()
                    if step % 10 == 0:
                        pbar.update(10)
                prey_mean_ratios.append(np.mean(prey_ratio))
                predator1_mean_ratios.append(np.mean(predator1_ratio))
                predator2_mean_ratios.append(np.mean(predator2_ratio))

                if simulation > 0:
                    max_cv_values.append(
                        max(
                            stats.variation(prey_mean_ratios),
                            stats.variation(predator1_mean_ratios),
                            stats.variation(predator2_mean_ratios)
                        ))
                    max_sem_values.append(
                        max(
                            stats.sem(prey_mean_ratios),
                            stats.sem(predator1_mean_ratios),
                            stats.sem(predator2_mean_ratios)
                        ))
            all_max_cv_values.append(max_cv_values)
            all_max_sem_values.append(max_sem_values)

    X = list(range(1, n_simulations))
    for i, errors in enumerate(all_max_cv_values):
        plt.plot(X, errors)

    plt.yscale('log')
    plt.xlabel('$n$ simulations')
    plt.ylabel('max $C_v$')
    plt.title('$C_v$ of ratio of total population for {} random initial conditions'.format(n_random_initial))
    plt.show()

    for i, errors in enumerate(all_max_sem_values):
        plt.plot(X, errors)

    plt.yscale('log')
    plt.xlabel('$n$ simulations')
    plt.ylabel(r'max $\sigma / \sqrt{n}$')
    plt.title('Standard error of ratio of total population for {} random initial conditions'.format(n_random_initial))
    plt.show()


def model_validation():
    initial_ratios = [
        [0.999, 0.001, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    T = 50
    count = []
    with tqdm(total=len(initial_ratios) * T) as pbar:
        for idx, ir in enumerate(initial_ratios):
            ca = Simple_CA3(0.9925629160207855, 0.154667180093502, 0.39186958820898643, 0.7644954722324826,
                            0.10463787447507766,
                            0.27740597266372047, 50, 50, bc.toroidal,
                            initial_ratio=ir)
            current_count = []
            for _ in range(T):
                current_count.append(ca.count[idx+1] / 2500)
                ca.step()
                pbar.update(1)
            count.append(current_count)
    X = list(range(T))
    plt.plot(X, count[0], label='Prey')
    plt.plot(X, count[1], label='Predator 1')
    plt.plot(X, count[2], label='Predator 2')
    plt.legend()
    plt.xlabel('Time steps')
    plt.ylabel('count / lattice size')
    plt.title('Model validation')
    plt.show()


if __name__ == '__main__':
    # animate_ca()
    # test()
    # error_plot()
    # error_plot_v2()
    # lattice_dimension()
    # sem_plot_bc()
    # lattice_size_error()
    # initial_ratio__error()
    model_validation()
