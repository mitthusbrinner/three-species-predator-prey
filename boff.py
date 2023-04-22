import math
from tqdm import tqdm


def prime_factor(n: int, p: int) -> int:
    """
    Find k such that p^k divides n where p is prime.
    """
    m = n
    k = 0
    while m % p == 0:
        m /= p
        k += 1
    return k


def period(n: int) -> int:
    """
    Finds k such that every kth number ending in n is also divisible by n

    We want to solve
        k * 10^d + n = m * n
    where d is the number of digits in n over integers k and m, so
        10^d * k / n = 5^d * 2^d * k / n
    must be an integer.
    """
    twos = prime_factor(n, 2)
    fives = prime_factor(n, 5)
    d = len(str(n))
    return n // (2 ** (min(d, twos)) * 5 ** (min(d, fives)))


def double(n, a, b):
    """
    So much pain.
    """
    d = len(str(n))
    per = period(n)
    t_first = math.ceil((a - n) / (per * 10 ** d))
    if (t_first * per) * (10 ** d) + n > b:
        return 0
    t_last = math.floor((b - n) / (per * 10 ** d))
    return t_last - t_first + 1 if t_last > t_first else 1


def num_ending(n, a, b):
    """
    Why are we still here, just to suffer?
    """
    d = len(str(n))
    m = a // (10 ** d)
    first_ending = m * 10 ** d + n if a % (10 ** d) <= n else (m + 1) * (10 ** d) + n
    if first_ending > b:  # to handle Hernadis paradox
        return 0

    m = b // (10 ** d)
    last_ending = m * 10 ** d + n if b % (10 ** d) >= n else (m - 1) * (10 ** d) + n
    return (last_ending - first_ending) // (10 ** d) + 1


def boff(n: int, a: int, b: int) -> int:
    """
    O(1) with respect to interval length a, b
    """
    multiples = b // n - (a - 1) // n
    ending = num_ending(n, a, b)
    double_counted = double(n, a, b)

    return multiples + ending - double_counted


def brute_force(n, a, b):
    d = len(str(n))
    total = 0
    mult = 0
    ending = 0
    double_counted = 0
    for i in range(a, b + 1):
        if i % 10 ** d == n or i % n == 0:
            total += 1
        if i % 10 ** d == n and i % n == 0:
            double_counted += 1
        if i % 10 ** d == n:
            ending += 1
        if i % n == 0:
            mult += 1
    return total, mult, ending, double_counted


def main():
    boff(3, 193, 230)
    return
    for a in tqdm(range(100, 200)):
        for b in range(a, a + 100):
            for n in range(2, 99):
                solution = boff(n, a, b)
                total, _, _, _ = brute_force(n, a, b)
                if total != solution:
                    print(n, a, b, 'solution', solution, 'exact', total)

    return

    n = 3
    A = [193]
    B = [230]

    for a in A:
        for b in B:
            solution = boff(n, a, b)
            total, mult, ending, double_counted = brute_force(n, a, b)
            print('--- exact')
            print('mult', mult)
            print('ending', ending)
            print('double_counted', double_counted)
            print('total', total)
            print('-----> CORRECT', solution == total)


if __name__ == '__main__':
    main()
