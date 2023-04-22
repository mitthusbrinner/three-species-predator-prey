import math


def main():
    n = 4
    k = 2

    for i in range(1, math.comb(n, k)):
        print('({} + 2m)'.format(math.comb(n, k) - i), end='')
    print()
    print('------------')
    for i in range(math.comb(n, k) - 1, 1 - 1, -1):
        print('{} * '.format(i), end='')


if __name__ == '__main__':
    main()
