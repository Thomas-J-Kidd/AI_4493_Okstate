import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.random import permutation


cities_2 = [[0.6606, 0.9500], [0.9695, 0.6740], [0.5906, 0.5029], [0.2124, 0.82740], [0.0398, 0.9697], [0.1367, 0.5979], [0.9536, 0.2184], [0.6091,0.7148],[0.8767, 0.2395], [0.8148, 0.2867], [0.3876,0.8200 ], [0.7041, 0.3296], [0.0231,0.1649], [0.3429, 0.3925], [0.7471, 0.8192], [0.5449, 0.9392], [0.9464,0.8191 ], [0.1247, 0.4351], [0.1636, 0.8646], [0.8668, 0.6768] ]
cities_x_1 = [0.6606, 0.9695, 0.5906, 0.2124, 0.0398, 0.1367, 0.9536, 0.6091, 0.8767, 0.8148, 0.3876, 0.7041, 0.0231, 0.3429, 0.7471, 0.5449, 0.9464, 0.1247, 0.1636, 0.8668]
cities_y_1 = [0.9500, 0.6740, 0.5029, 0.82740, 0.9697, 0.5979, 0.2184, 0.7148, 0.2395, 0.2867,  0.8200, 0.3296, 0.1649, 0.3925, 0.8192, 0.9392, 0.8191, 0.4351, 0.8646, 0.6768]

cities = [[41, 49], [35, 17], [55, 45], [55, 20], [15, 30], [25, 30], [20, 50], [10, 43] ,[55, 60],[30, 60],[20, 65],[50, 35],[30, 25],[15, 10],[30, 5],[10, 20],[5, 30],[20, 40],[15, 60],[45, 65],[45, 20],[45, 10],[55, 5],[65, 35],[65, 20],[45, 30],[35,40],[41, 37],[64,42],[40,60],[31,52],[35,69],[53,52],[65,55],[63,65],[2,60],[30,30],[5,5],[60,12],[40,25],[42,7],[24,12],[23,3],[11,14],[6,38],[2,48],[8,56],[13,52],[6,68],[47,47],[49,58],[27,43],[37,31],[57,29],[63,23],[53,12],[32,12],[36,26],[21,24],[17,34],[12,24],[24,58],[27,69],[15,77],[62,77],[49,73],[67,5],[56,39],[37,47],[37,56],[57,68],[47,16],[44,17],[46,13],[49,11],[49,42],[53,42],[61,52],[57,48],[56,37],[55,54],[15,47],[14,37],[11,31],[16,22],[4,18],[28,18],[26,52],[26,35],[31,67],[15,19],[22,22],[18,24],[26,27],[25,24],[22,27],[25,21],[19,21],[20,26],[18,18],[35,35]]


distance_matrix=[[0 for j in range(18)] for i in range(18)]

for i in range(0, 18):
    for j in range(0, 18):
        distance_matrix[i][j] = np.sqrt((cities_x_1[j] - cities_x_1[i] )**2+(cities_y_1[j] - cities_y_1[i] )**2)



class Simulated_Annealing:

    def __init__(self, route, temperature, alpha, iterations):
        self.route = route
        self.temperature = temperature
        self.alpha = alpha
        self.iterations = iterations
        print("initalized the testing values for simulated annealing on the base route given")

    
    def get_distance(self, route):
        distance = 0
        for i in range(100):
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

            for j in range(5):
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
iterations = 2000
alpha = 0.5
temperature = 200

obj = Simulated_Annealing(cities, temperature, alpha, iterations)

scores, temps, best_scores ,best_route = obj.run()

df_original = pd.DataFrame(cities, columns=["x", "y"])
df_best = pd.DataFrame(best_route, columns=["x", "y"])

plt.plot(scores, c = "Green")
plt.plot(best_scores, c = "Red")
plt.plot(temps, c="Black")
plt.show()

plt.scatter(df_original["x"], df_original["y"], label="original data", c="Red")
plt.plot(df_best["x"], df_best["y"], label="best data", c="Blue")
plt.show()
