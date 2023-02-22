import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.random import permutation



def next_solution_in_range(df):

    var1 = random.randrange(0,19)
    var2 = random.randrange(0,19)

    #print("\n\n", var1)
    #print(var2, "\n\n")

    new_df = df.copy()

    temp1 = new_df.iloc[var1]
    temp2 = new_df.iloc[var2]

    new_df.iloc[var1] = temp2
    new_df.iloc[var2] = temp1

    new_df = new_df.reset_index(drop = True)

    #print(df)
    #print(new_df)
    return new_df
        
def temp_change(temp, a):
    return a*temp

    
def calc_distance_of_df(df):
    distance = 0.0
    for i in range(19):
        distance += np.sqrt((df["x"].loc[idx[i+1]]-df["x"].loc[idx[i]])**2+(df["y"].loc[idx[i+1]]-df["y"].loc[idx[i]])**2)
    #print("The distance of the given df: %f", distance)
    return distance

def check_acceptance(temp, new_solution, current_solution):
    prob = round( np.exp(-(new_solution-current_solution)/temp), 3)
    if prob < random.uniform(0, 1):
        return True
    else:
        return False


def swap_elements(df):
    df_new = df.copy()
        #df_new = df.sample(frac=1).reset_index()
        #swap_list_indx = range(1, len(df) - 1)

    shuffled = df_new.iloc[permutation(df_new.index)]
    shuffled = shuffled.reset_index(drop = True)
        #print(shuffled)
        
        # i = random.randint(swap_list_indx[0], swap_list_indx[-1])
        # j = random.randint(swap_list_indx[0], swap_list_indx[-1])
        # print("we have: %i and %i", i, j)

        # if i == j:
        #     while i == j:
        #         j = random.randint(swap_list_indx[0], swap_list_indx[-1])

        
    return shuffled

    



cities_x = [0.6606, 0.9695, 0.5906, 0.2124, 0.0398, 0.1367, 0.9536, 0.6091, 0.8767, 0.8148, 0.9500, 0.6740, 0.5029, 0.82740, 0.9697, 0.5979, 0.2184, 0.7148, 0.2395, 0.2867]
cities_y = [0.3876, 0.7041, 0.0231, 0.3429, 0.7471, 0.5449, 0.9464, 0.1247, 0.1636, 0.8668, 0.8200, 0.3296, 0.1649, 0.3925, 0.8192, 0.9392, 0.8191, 0.4351, 0.8646, 0.6768]
cities = [[0.6606, 0.3876], [0.9695, 0.7041], [0.5906, 0.0231], [0.2124,0.3429], [0.0398, 0.7471], [0.1367, 0.5449], [0.9536, 0.9464], [0.6091, 0.1247], [0.8767, 0.1636], [0.8148, 0.8668], [0.9500, 0.8200], [0.6740, 0.3296], [0.5029, 0.1649], [0.82740, 0.3925], [0.9697, 0.8192], [0.5979, 0.9392], [0.2184, 0.8191], [0.7148, 0.4351], [0.2395, 0.8646], [0.2867, 0.6768]]

idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


df_original = pd.DataFrame(cities,index=idx, columns=["x", "y"])





# generate an initial random solution

#random_start = df_original.sample(frac=1).reset_index()


def SA(df, iterations, t, a):

    scores = []
    best_scores = []
    temps = []
    count = 0

    random_start = df.sample(frac=1).reset_index()
    best_df = random_start
    current_distance = calc_distance_of_df(random_start)
    best_distance = calc_distance_of_df(random_start)

    for i in range(iterations):
        df_new = next_solution_in_range(best_df)
        new_distance = calc_distance_of_df(df_new)
        scores.append(new_distance)
        


        if new_distance < best_distance:
            best_df = df_new.copy()
            best = new_distance.copy()
            best_scores.append(best)
            count += 1


        if check_acceptance(t, new_distance, current_distance):
            best_df = df_new.copy()
            current_distance = new_distance.copy()
            
        temps.append(t)
        t = temp_change(t, a)
        df_new = df_new.reset_index(drop=True)

    print("Number of times the new distance was less then the best distance: %i" ,count)

    return scores, best_scores, temps, best_df, random_start


def two_opt(route, dist_df):
    """Apply the 2-opt algorithm to improve a route.

    Args:
        route (list): A list of node indices representing the route.
        dist_df (Pandas DataFrame): A DataFrame representing the distances between
                                    each pair of nodes.

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
                new_dist = calculate_distance(new_route, dist_df)
                if new_dist < calculate_distance(route, dist_df):
                    route = new_route
                    improved = True

    return route


def calculate_distance(route, dist_df):
    """Calculate the total distance of a route.

    Args:
        route (list): A list of node indices representing the route.
        dist_df (Pandas DataFrame): A DataFrame representing the distances between
                                     each pair of nodes.

    Returns:
        float: The total distance of the route.
    """
    distance = 0
    for i in range(len(route) - 1):
        distance += dist_df.loc[route[i], route[i + 1]]
    distance += dist_df.loc[route[-1], route[0]]
    return distance



iter = 1000
t=3
a = 0.99
scores, best_scores, temps, best_df, random_start = SA(df_original, iter, t, a)


plt.plot(scores, c = "Green")
plt.plot(best_scores, c = "Red")
plt.plot(temps, c="Black")
plt.show()

# plt.scatter(df_original["x"], df_original["y"], label="original data", c="Red")
# plt.plot(random_start["x"], random_start["y"],label="first random data", c="Blue" )
# plt.show()

# plt.scatter(df_original["x"], df_original["y"], label="original data", c="Red")
# plt.plot(best_df["x"], best_df["y"],label="first random data", c="Green" )
# plt.show()
