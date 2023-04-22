import random
import json

from Simple_CA3 import Simple_CA3
from Predator_Prey_CA import bc


def random_search(N_combinations: int, T: int):
    combinations = []
    for _ in range(N_combinations):
        combination = []
        for _ in range(5 + 1):
            combination.append(random.uniform(0, 1))
        combinations.append(combination)

    non_zero_z = []
    for idx, combination in enumerate(combinations):
        if idx % 10 == 0:
            print('{0:.2f}%'.format((idx + 1) / N_combinations * 100))
        ca = Simple_CA3(
            combination[0], combination[1], combination[2], combination[3], combination[4], combination[5],
            50, 50, bc.toroidal, initial_ratio=[0.901, 0.033, 0.033, 0.033])
        z = 0
        for _ in range(T):
            total_population = ca.tot_population
            if total_population > 0:
                alpha = ca.count[1] / total_population  # prey ratio
                beta = ca.count[2] / total_population  # predator 1 ratio
                gamma = ca.count[3] / total_population  # predator 2 ratio
                z = alpha * beta * gamma
                if z == 0:  # if one species is extinct, move on to the next combination
                    break
            else:
                break  # in actors left, move on to the next combination
            ca.step()
        if z != 0:  # only append if z is not zero
            normed_data = (combination, z * 27)
            non_zero_z.append(normed_data)
    return non_zero_z


def main():
    non_zero_z = random_search(300, 1000)

    with open('data.json') as f:
        stored_data = [tuple(x) for x in json.load(f)]
        print('stored data')
    for idx, data in enumerate(stored_data):
        print(str(idx + 1) + ': \t' + str(data))

    stored_data.extend(non_zero_z)
    stored_data.sort(key=lambda tup: tup[1], reverse=True)

    with open('data.json', 'w') as f:
        json.dump(stored_data, f)
    print('new data')
    for idx, data in enumerate(stored_data):
        print(str(idx + 1) + ': \t' + str(data))


if __name__ == '__main__':
    """
    N_combinations = 10

    combinations = []
    for _ in range(N_combinations):
        combination = []
        for _ in range(5 + 1):
            combination.append(random.uniform(0, 1))
        combinations.append(combination)

    # print(combinations)

    hull = ConvexHull(combinations)
    print(hull.simplices)
    """

    main()
