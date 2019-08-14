import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA


def SK(x, y, v, variogram, grid):
    cov_angulos = np.zeros((x.shape[0], x.shape[0]))
    cov_distancias = np.zeros((x.shape[0], x.shape[0]))
    K = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0] - 1):
        cov_angulos[i, i:] = np.arctan2((y[i:] - y[i]), (x[i:] - x[i]))
        cov_distancias[i, i:] = np.sqrt((x[i:] - x[i]) ** 2 + (y[i:] - y[i]) ** 2)

    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if cov_distancias[i, j] != 0:
                amp = np.sqrt(
                    (variogram[1] * np.cos(cov_angulos[i, j])) ** 2 + (variogram[0] * np.sin(cov_angulos[i, j])) ** 2)
                K[i, j] = v[:].var() * (1 - np.e ** (-3 * cov_distancias[i, j] / amp))
    K = K + K.T

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            distancias = np.sqrt((i - x[:]) ** 2 + (j - y[:]) ** 2)
            angulos = np.arctan2(i - y[:], j - x[:])
            amplitudes = np.sqrt((variogram[1] * np.cos(angulos[:])) ** 2 + (variogram[0] * np.sin(angulos[:])) ** 2)
            M = v[:].var() * (1 - np.e ** (-3 * distancias[:] / amplitudes[:]))
            W = LA.solve(K, M)
            grid[i, j] = np.sum(W * (v[:] - v[:].mean())) + v[:].mean()
    return grid


def OK(x, y, v, variogram, grid):
    cov_angulos = np.zeros((x.shape[0], x.shape[0]))
    cov_distancias = np.zeros((x.shape[0], x.shape[0]))
    K = np.zeros((x.shape[0] + 1, x.shape[0] + 1))

    for i in range(x.shape[0] - 1):
        cov_angulos[i, i:] = np.arctan2((y[i:] - y[i]), (x[i:] - x[i]))
        cov_distancias[i, i:] = np.sqrt((x[i:] - x[i]) ** 2 + (y[i:] - y[i]) ** 2)

    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if cov_distancias[i, j] != 0:
                amp = np.sqrt(
                    (variogram[1] * np.cos(cov_angulos[i, j])) ** 2 + (variogram[0] * np.sin(cov_angulos[i, j])) ** 2)
                K[i, j] = v[:].var() * (1 - np.e ** (-3 * cov_distancias[i, j] / amp))
    K = K + K.T
    K[-1, :] = 1
    K[:, -1] = 1
    K[-1, -1] = 0

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            distancias = np.sqrt((i - x[:]) ** 2 + (j - y[:]) ** 2)
            angulos = np.arctan2(i - y[:], j - x[:])
            amplitudes = np.sqrt((variogram[1] * np.cos(angulos[:])) ** 2 + (variogram[0] * np.sin(angulos[:])) ** 2)
            M = np.ones((x.shape[0] + 1, 1))
            M[:-1, 0] = v[:].var() * (1 - np.e ** (-3 * distancias[:] / amplitudes[:]))
            W = LA.solve(K, M)
            grid[i, j] = np.sum(W[:-1, 0] * (v[:]))
    return grid


def plot_grid(x, y, v, grid):
    plt.imshow(grid.T, origin='lower', interpolation='nearest', cmap='jet')
    plt.scatter(x, y, c=v, cmap='jet', s=120)
#     plt.contour(grid.T, color='k')
    plt.xlim(0, grid.shape[0])
    plt.ylim(0, grid.shape[1])
    plt.colorbar()
    plt.grid()
    plt.show()
