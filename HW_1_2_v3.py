import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.random import permutation

cities = [[41, 49], [35, 17], [55, 45], [55, 20], [15, 30], [25, 30], [20, 50], [10, 43] ,[55, 60],[30, 60],[20, 65],[50, 35],[30, 25],[15, 10],[30, 5],[10, 20],[5, 30],[20, 40],[15, 60],[45, 65],[45, 20],[45, 10],[55, 5],[65, 35],[65, 20],[45, 30],[35,40],[41, 37],[64,42],[40,60],[31,52],[35,69],[53,52],[65,55],[63,65],[2,60],[30,30],[5,5],[60,12],[40,25],[42,7],[24,12],[23,3],[11,14],[6,38],[2,48],[8,56],[13,52],[6,68],[47,47],[49,58],[27,43],[37,31],[57,29],[63,23],[53,12],[32,12],[36,26],[21,24],[17,34],[12,24],[24,58],[27,69],[15,77],[62,77],[49,73],[67,5],[56,39],[37,47],[37,56],[57,68],[47,16],[44,17],[46,13],[49,11],[49,42],[53,42],[61,52],[57,48],[56,37],[55,54],[15,47],[14,37],[11,31],[16,22],[4,18],[28,18],[26,52],[26,35],[31,67],[15,19],[22,22],[18,24],[26,27],[25,24],[22,27],[25,21],[19,21],[20,26],[18,18],[35,35]]

#print(cities[62])







def two_opt(route, temp, a):
    """Apply the 2-opt algorithm to improve a route.

    Args:
        route (list): A list of node indices representing the route.
        dist_matrix (list of lists): A distance matrix representing the distances
                                     between each pair of nodes.

    Returns:
        list: The improved route.
    """
    scores = []
    best_scores = []
    temps = []
    n = len(route)
    improved = True
    count = 0
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                count +=1
                if j - i == 1:
                    continue
                new_route = route[:]
                x = random.randint(-1,1)
                y = random.randint(-1,1)
                new_route
                new_route[i:j] = route[j - x:i - y:-1]
                current_dist = calculate_distance(route)
                new_dist = calculate_distance(new_route)
                prob = np.exp(-(new_dist-current_dist)/temp)
                scores.append(new_dist)
                if new_dist < calculate_distance(route):
                    temp = temp*a
                    route = new_route
                    improved = True
                    best_scores.append(new_dist)
                    temps.append(temp)
                    
                elif prob < random.uniform(0, 1):
                    route = new_route
                    improved = True
                else:
                    route = route
    
    #print("The count: ", count)

    return route, scores, best_scores, temps


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
    for i in range(len(route)-1):
        distance  += np.sqrt((route[i+1][0] - route[i][0])**2+(route[i+1][1] - route[i][1])**2)
    print(distance)
    return distance


t=100
alpha = 0.8

best_route, scores, best_scores, temps = two_opt(cities, t, alpha)
best_route.append(best_route[0])


df_original = pd.DataFrame(cities, columns=["x", "y"])
df_best = pd.DataFrame(best_route, columns=["x", "y"])

plt.plot(scores, c = "Green")
plt.plot(best_scores, c = "Red")
plt.plot(temps, c="Black")
plt.show()

plt.scatter(df_original["x"], df_original["y"], label="original data", c="Red")
plt.plot(df_best["x"], df_best["y"], label="best data", c="Blue")
plt.show()