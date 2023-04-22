import math
from abc import ABC, abstractmethod
from enum import Enum
import random


class bc(Enum):
    toroidal = 0
    reflective = 1


class Predator_Prey_CA(ABC):
    """
    This (abstract) class includes the basic machinery you may need to implement a predator-prey cellular automaton,
    just implement the rules for what to do at each position in '_apply_rules' and voilà!
    """

    def __init__(self, n_actors: int, height: int, width: int,
                 boundary_conditions: bc, initial_conditions=None,
                 initial_ratio=None):
        """
        :param n_actors: Number of prey/predators in the ca
        :param height: Height of the geometry
        :param width: Width of the geometry
        :param boundary_conditions: Either toroidal (periodic) or reflective
        :param initial_conditions: Initial state of the lattice, if not set, each position is randomized
        """
        self.n_actors = n_actors
        self.height = height
        self.width = width
        self.boundary_conditions = boundary_conditions
        self.lattice = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if initial_conditions is None and initial_ratio is None:
            self._randomize_lattice()
        elif initial_conditions is not None and initial_ratio is None:
            self.lattice = initial_conditions
        elif initial_conditions is None and initial_ratio is not None:
            self._initialise_with_ratios(initial_ratio)

        self.count = [0] * (n_actors + 1)
        self.tot_population = 0
        self._update_count()

    def step(self):
        """
        Advances the cellular automaton one step using the rules defined in '_apply_rules'
        """
        new_lattice = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                new_lattice[y][x] = self._apply_rules((y, x))
        self.lattice = new_lattice
        self._update_count()

    @abstractmethod
    def _apply_rules(self, position) -> int:
        """
        The rules of the cellular automaton, may consist of multiple phases.
        """
        pass

    def _randomize_lattice(self) -> None:
        """
        Fills the lattice with random actors.
        """
        states = list(range(self.n_actors + 1))
        for y in range(self.height):
            for x in range(self.width):
                self.lattice[y][x] = random.choice(states)

    def _initialise_with_ratios(self, ratios) -> None:
        states = []
        total_positions = self.width * self.height
        for i in range(self.n_actors + 1):
            states += [i] * math.ceil(ratios[i] * total_positions)

        if len(states) != total_positions:
            states = states[:total_positions]
        random.shuffle(states)

        idx = 0
        for y in range(self.height):
            for x in range(self.width):
                if idx >= total_positions:
                    continue
                self.lattice[y][x] = states[idx]
                idx += 1
            idx += 1

    def _update_count(self) -> None:
        """
        Counts the number of each actor that is in the lattice.
        """
        self.count = [0] * (self.n_actors + 1)
        for y in range(self.height):
            for x in range(self.width):
                state = self.lattice[y][x]
                self.count[state] += 1
        self.tot_population = sum(self.count[1:])

    def _moore_neighbourhood(self, position, radius: int):
        """
        Finds the number of predators and prey within a Moore neighbourhood of radius 'radius' of
        position.
        """
        count = [0] * (self.n_actors + 1)
        y_pos, x_pos = position
        for y in range(y_pos - radius, y_pos + radius + 1):
            for x in range(x_pos - radius, x_pos + radius + 1):
                if y == y_pos and x == x_pos:
                    continue
                if self.boundary_conditions == bc.toroidal:
                    y = y % self.height
                    x = x % self.width
                elif self.boundary_conditions == bc.reflective:
                    if not (0 < y < self.height) or not (0 < x < self.width):
                        continue
                state = self.lattice[y][x]
                count[state] += 1
                # self.lattice[y][x] = 1  # for test
        return count

    def _von_neumann(self, position, radius: int):
        """
        Finds the number of predators and prey within a von Neumann neighbourhood of radius 'radius' of
        position.
        """
        count = [0] * (self.n_actors + 1)
        y_pos, x_pos = position
        widths = list(range(0, radius + 1)) + list(range(radius - 1, 0 - 1, -1))
        for i, y in enumerate(range(y_pos - radius, y_pos + radius + 1)):
            for x in range(x_pos - widths[i], x_pos + widths[i] + 1):
                if y == y_pos and x == x_pos:
                    continue
                if self.boundary_conditions == bc.toroidal:
                    y = y % self.height
                    x = x % self.width
                elif self.boundary_conditions == bc.reflective:
                    if not (0 < y < self.height) or not (0 < x < self.width):
                        continue
                state = self.lattice[y][x]
                count[state] += 1
                # self.lattice[y][x] = 1  # for test
        return count
