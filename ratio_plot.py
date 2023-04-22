import json
import numpy as np

from matplotlib import pyplot as plt, animation


def plane_fit(points: list[tuple]) -> tuple:
    """
    Finds the plane of best fit to a set of points in 3d using its SVD.

    Returns a, b, c, d i n ax + by +cz = d

    https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    """
    X, Y, Z = zip(*points)
    matrix = [X, Y, Z]
    centroid = np.mean(matrix, axis=1, keepdims=True)
    svd = np.linalg.svd(matrix - centroid)
    left = svd[0]
    normal = left[:, -1]
    a = normal[0]
    b = normal[1]
    c = normal[2]
    d = float(a * centroid[0] + b * centroid[1] + c * centroid[2])

    return a, b, c, d


def main():
    with open('data10000.json') as f:
        stored_data = [tuple(x) for x in json.load(f)]
        print('Loaded', len(stored_data), 'datasets')

    bin99 = []
    bin90 = []
    bin80 = []
    bin0 = []
    all_points = []
    all_z_values = []

    for data in stored_data:
        combinations, z_normed = data
        prey_ratio = combinations[1] / combinations[0]
        predator1_ratio = combinations[3] / combinations[2]
        predator2_ratio = combinations[5] / combinations[4]

        point = (prey_ratio, predator1_ratio, predator2_ratio)
        if z_normed > 0:
            all_z_values.append(z_normed)
            all_points.append(point)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(*zip(*all_points), c=all_z_values, cmap='plasma')
    fig.colorbar(img)
    plt.title('Random Search Result')
    ax.set_xlabel(r'$P_B / P_D$')
    ax.set_ylabel(r'$T_B / T_D$')
    ax.set_zlabel(r'$H_B / H_D$')

    # do fit
    xs, ys, zs = zip(*all_points)
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X, Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                       np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fit[0] * X[i, j] + fit[1] * Y[i, j] + fit[2]
    # ax.plot_surface(X, Y, Z, color='k', alpha=.5)

    print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    # print("errors:")
    # print(errors)
    print("residual: {}".format(residual))

    def rotate(angle):
        ax.view_init(azim=angle)

    '''
    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save('test.gif', writer=animation.PillowWriter(fps=20))
    '''

    plt.show()


if __name__ == '__main__':
    main()
