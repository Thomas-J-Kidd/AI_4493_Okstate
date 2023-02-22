import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.random import permutation


cities_x = [0.6606, 0.9695, 0.5906, 0.2124, 0.0398, 0.1367, 0.9536, 0.6091, 0.8767, 0.8148, 0.9500, 0.6740, 0.5029, 0.82740, 0.9697, 0.5979, 0.2184, 0.7148, 0.2395, 0.2867]
cities_y = [0.3876, 0.7041, 0.0231, 0.3429, 0.7471, 0.5449, 0.9464, 0.1247, 0.1636, 0.8668, 0.8200, 0.3296, 0.1649, 0.3925, 0.8192, 0.9392, 0.8191, 0.4351, 0.8646, 0.6768]
cities = [[0.6606, 0.3876], [0.9695, 0.7041], [0.5906, 0.0231], [0.2124,0.3429], [0.0398, 0.7471], [0.1367, 0.5449], [0.9536, 0.9464], [0.6091, 0.1247], [0.8767, 0.1636], [0.8148, 0.8668], [0.9500, 0.8200], [0.6740, 0.3296], [0.5029, 0.1649], [0.82740, 0.3925], [0.9697, 0.8192], [0.5979, 0.9392], [0.2184, 0.8191], [0.7148, 0.4351], [0.2395, 0.8646], [0.2867, 0.6768]]

cities_x_1 = np.array([0.6606, 0.9695, 0.5906, 0.2124, 0.0398, 0.1367, 0.9536, 0.6091, 0.8767, 0.8148, 0.3876, 0.7041, 0.0231, 0.3429, 0.7471, 0.5449, 0.9464, 0.1247, 0.1636, 0.8668])
cities_y_1 = np.array([0.9500, 0.6740, 0.5029, 0.82740, 0.9697, 0.5979, 0.2184, 0.7148, 0.2395, 0.2867,  0.8200, 0.3296, 0.1649, 0.3925, 0.8192, 0.9392, 0.8191, 0.4351, 0.8646, 0.6768])
cities_1 = np.vstack((cities_x_1, cities_y_1))
cities_2 = [[0.6606, 0.9500], [0.9695, 0.6740], [0.5906, 0.5029], [0.2124, 0.82740], [0.0398, 0.9697], [0.1367, 0.5979], [0.9536, 0.2184], [0.6091,0.7148],[0.8767, 0.2395], [0.8148, 0.2867], [0.3876,0.8200 ], [0.7041, 0.3296], [0.0231,0.1649], [0.3429, 0.3925], [0.7471, 0.8192], [0.5449, 0.9392], [0.9464,0.8191 ], [0.1247, 0.4351], [0.1636, 0.8646], [0.8668, 0.6768] ]

idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

dist_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, ]

for i in range(19):
    dist_list[i] = np.sqrt((cities_x[i+1] - cities_x[i])**2+(cities_y[i+1] - cities_y[i])**2)



def two_opt(route):
    """Apply the 2-opt algorithm to improve a route.

    Args:
        route (list): A list of node indices representing the route.
        dist_matrix (list of lists): A distance matrix representing the distances
                                     between each pair of nodes.

    Returns:
        list: The improved route.
    """
    n = len(route)
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]
                new_dist = calculate_distance(new_route)
                if new_dist < calculate_distance(route):
                    route = new_route
                    improved = True

    return route


def calculate_distance(route):
    """Calculate the total distance of a route.

    Args:
        route (list): A list of node indices representing the route.
        dist_matrix (list of lists): A distance matrix representing the distances
                                     between each pair of nodes.

    Returns:
        float: The total distance of the route.
    """
    distance = 0
    for i in range(19):
        distance  += np.sqrt((route[i+1][0] - route[i][0])**2+(route[i+1][1] - route[i][1])**2)
    print(distance)
    return distance



best_route = two_opt(cities_2)
best_route.append(best_route[0])


df_original = pd.DataFrame(cities_2, columns=["x", "y"])
df_best = pd.DataFrame(best_route, columns=["x", "y"])



plt.scatter(df_original["x"], df_original["y"], label="original data", c="Red")
plt.plot(df_best["x"], df_best["y"], label="best data", c="Blue")
plt.show()