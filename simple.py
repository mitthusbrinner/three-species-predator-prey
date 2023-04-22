import random

import numpy as np
from matplotlib import pyplot as plt, animation


class simple_CA:
    def __init__(self,
                 p_pred_death: float,
                 p_pred_birth: float,
                 p_prey_death: float,
                 p_prey_birth: float,
                 width: int,
                 height: int):
        self.height = height
        self.width = width
        self.p_prey_birth = p_prey_birth
        self.p_prey_death = p_prey_death
        self.p_pred_birth = p_pred_birth
        self.p_pred_death = p_pred_death

        self.n_pred = 0
        self.n_prey = 0
        self.total = 0

        self.lattice = [[0 for _ in range(self.width)] for _ in range(self.height)]

        # random initial condition
        for y in range(self.height):
            for x in range(self.width):
                self.lattice[y][x] = random.choice([0, 0, 1, 2])  # 0 empty, 1 prey, 2 predator

        self._update_count()

    def step(self):
        for y in range(self.height):
            for x in range(self.width):
                self.lattice[y][x] = self._apply_rules((y, x))

    def _apply_rules(self, position: tuple[int, int]) -> int:
        state = self.lattice[position[1]][position[0]]
        if state == 0:  # empty
            n_pred, n_prey = self._n_pred_prey_moore_neighbourhood(position)
            if n_prey == 0 or n_pred > 0:
                return 0  # do nothing
            else:
                p = random.uniform(0, 1)
                if p < (1 - self.p_prey_birth) ** n_prey:
                    return 1  # breeding!
                return 0
        elif state == 1:  # prey
            p = random.uniform(0, 1)
            n_pred, _ = self._n_pred_prey_moore_neighbourhood(position)
            if p < (1 - self.p_prey_death) ** n_pred:
                return 1  # hunt failed
            else:
                r = random.uniform(0, 1)
                if r < self.p_pred_birth:
                    return 2  # cell becomes predator by breeding
                return 0
        elif state == 2:  # predator
            p = random.uniform(0, 1)
            if p < self.p_pred_death:
                return 0
            else:
                return 2

    def _n_pred_prey_moore_neighbourhood(self, position: tuple[int, int]) -> tuple[int, int]:
        n_pred = 0
        n_prey = 0
        relative_positions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        # print(position)
        for relative_position in relative_positions:
            absolute_pos = np.add(position, relative_position)
            # absolute_pos = (absolute_pos[1] % (self.height - 1), absolute_pos[0] % (self.width - 1))  # periodic BC
            if absolute_pos[0] > self.height - 1 or absolute_pos[0] < 0 or \
                    absolute_pos[1] > self.width - 1 or absolute_pos[1] < 0:  # reflective BC
                # print('outside')
                continue
            # print(absolute_pos)
            state = self.lattice[absolute_pos[1]][absolute_pos[0]]
            if state == 1:
                n_prey += 1
            elif state == 2:
                n_pred += 1
        # print('-----')
        # self._update_count()
        return n_pred, n_prey

    def _update_count(self):
        self.n_prey = 0
        self.n_pred = 0
        for y in range(self.height):
            for x in range(self.width):
                state = self.lattice[y][x]
                if state == 1:
                    self.n_prey += 1
                elif state == 2:
                    self.n_pred += 1
        self.total = self.n_prey + self.n_pred

def main():
    ca = simple_CA(0.2, 0.8, 0.8, 0.8, 50, 50)
    steps = 300
    """
    step_n_pred = []
    step_n_prey = []
    for n in range(steps):
        step_n_pred.append((n, ca.n_pred/ca.total))
        step_n_prey.append((n, ca.n_prey/ca.total))
        ca.step()
    plt.plot(*zip(*step_n_pred), label='predator')
    plt.plot(*zip(*step_n_prey), label='prey')
    plt.legend()
    plt.show()
        """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(ca.lattice, cmap=None, interpolation='nearest')
    fig.colorbar(im)


    def update(step):
        ca.step()
        im.set_array(ca.lattice)
        return im,

    anim = animation.FuncAnimation(fig, update, frames=200, interval=10, blit=True, repeat=False)
    plt.show()


if __name__ == '__main__':
    main()
