import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.random import permutation


cities_2 = [[0.6606, 0.9500], [0.9695, 0.6740], [0.5906, 0.5029], [0.2124, 0.82740], [0.0398, 0.9697], [0.1367, 0.5979], [0.9536, 0.2184], [0.6091,0.7148],[0.8767, 0.2395], [0.8148, 0.2867], [0.3876,0.8200 ], [0.7041, 0.3296], [0.0231,0.1649], [0.3429, 0.3925], [0.7471, 0.8192], [0.5449, 0.9392], [0.9464,0.8191 ], [0.1247, 0.4351], [0.1636, 0.8646], [0.8668, 0.6768] ]


class Simulated_Annealing:

    def __init__(self, route, temperature, alpha, iterations):
        self.route = route
        self.temperature = temperature
        self.alpha = alpha
        self.iterations = iterations
        print("initalized the testing values for simulated annealing on the base route given")

    
    def get_distance(self, route):
        distance = 0
        for i in range(19):
            distance  += np.sqrt((route[i+1][0] - route[i][0])**2+(route[i+1][1] - route[i][1])**2)
        #print(distance)
        return distance

    def get_acceptance(self, new_d, current_d, temp):
        prob = np.exp(-(new_d-current_d)/temp)
        if prob < random.uniform(0, 1):
            return True
        else:
            return False

    def set_temp(self, alpha, temp):
        return alpha*temp

    def get_next_solution(self, route):
    
        i, j = random.sample(range(len(route)), 2)
        new_route = route.copy()
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route
        


    
    def run(self):

        temperature = self.temperature
        alpha = self.alpha
        iterations = self.iterations
        route = self.route

        scores = []
        best_scores = []
        temps = []


        for i in range(iterations):

            new_route = self.get_next_solution(route)
            new_dist = self.get_distance(new_route)
            current_dist = self.get_distance(route)
        
            if new_dist < current_dist:
                route = new_route
                best_route = new_route
                best_scores.append(self.get_distance(best_route))

            elif self.get_acceptance(new_dist, current_dist, temperature):
                route = new_route
                
            scores.append(new_dist)
            temperature = self.set_temp(alpha, temperature)
            temps.append(temperature)


        return scores, temps, best_scores, best_route




cities_2 = [[0.6606, 0.9500], [0.9695, 0.6740], [0.5906, 0.5029], [0.2124, 0.82740], [0.0398, 0.9697], [0.1367, 0.5979], [0.9536, 0.2184], [0.6091,0.7148],[0.8767, 0.2395], [0.8148, 0.2867], [0.3876,0.8200 ], [0.7041, 0.3296], [0.0231,0.1649], [0.3429, 0.3925], [0.7471, 0.8192], [0.5449, 0.9392], [0.9464,0.8191 ], [0.1247, 0.4351], [0.1636, 0.8646], [0.8668, 0.6768] ]
iterations = 1000
alpha = 0.99
temperature = 2

obj = Simulated_Annealing(cities_2, temperature, alpha, iterations)

scores, temps, best_scores ,best_route = obj.run()

df_original = pd.DataFrame(cities_2, columns=["x", "y"])
df_best = pd.DataFrame(best_route, columns=["x", "y"])

plt.plot(scores, c = "Green")
plt.plot(best_scores, c = "Red")
plt.plot(temps, c="Black")
plt.show()

plt.scatter(df_original["x"], df_original["y"], label="original data", c="Red")
plt.plot(df_best["x"], df_best["y"], label="best data", c="Blue")
plt.show()
