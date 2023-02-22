import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random




class SA:

    def __init__(self, iterations, temp, df, a):
        self.M = iterations  # iterations
        self.T = temp  # temperature
        self.df = df  # dataframe
        self.a = a  # gamma value
        self.idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]



    def next_solution_in_range(self, df):
        new_df = df.sample(frac=1).reset_index()
        return new_df
        
    def temp_change(T, a):
        return a*T

    
    def calc_distance_of_df(self, df):
        distance = 0.0
        print(df["x"])
        for i in range(19):
            distance += np.sqrt((df["x"].loc[idx[i+1]]-df["x"].loc[idx[i]])**2+(df["y"].loc[idx[i+1]]-df["y"].loc[idx[i]])**2)
        print("The distance of the given df: %f", distance)
        return distance

    def check_acceptance(self, T, new_solution, current_solution):
        prob = np.exp(-(new_solution-current_solution)/T)
        if prob < random.uniform(0, 1):
            return True
        else:
            return False

    def run(self):

        T = self.T
        a = self.a
        df = self.df
        
        scores = []
        best_scores = []
        temps = []

        current_distance = self.calc_distance_of_df(self.df)
        best_distance = self.calc_distance_of_df(self.df)

        for i in range(self.M):
            df_new = self.next_solution_in_range(df)

            new_distance = self.calc_distance_of_df(df_new)
            scores.append(new_distance)

        
            if new_distance < best_distance:
                best_df = df_new.copy()
                best = new_distance.copy()

            best_scores.append(best)

            if self.check_acceptance(T, new_distance, current_distance):
                df = df_new.copy()
                current_distance = new_distance.copy()
            
            temps.append(T)
            T = self.temp_change(T, a)

        return scores, best_scores, temps, best_df




    



cities_x = [0.6606, 0.9695, 0.5906, 0.2124, 0.0398, 0.1367, 0.9536, 0.6091, 0.8767, 0.8148, 0.9500, 0.6740, 0.5029, 0.82740, 0.9697, 0.5979, 0.2184, 0.7148, 0.2395, 0.2867]
cities_y = [0.3876, 0.7041, 0.0231, 0.3429, 0.7471, 0.5449, 0.9464, 0.1247, 0.1636, 0.8668, 0.8200, 0.3296, 0.1649, 0.3925, 0.8192, 0.9392, 0.8191, 0.4351, 0.8646, 0.6768]
cities = [[0.6606, 0.3876], [0.9695, 0.7041], [0.5906, 0.0231], [0.2124,0.3429], [0.0398, 0.7471], [0.1367, 0.5449], [0.9536, 0.9464], [0.6091, 0.1247], [0.8767, 0.1636], [0.8148, 0.8668], [0.9500, 0.8200], [0.6740, 0.3296], [0.5029, 0.1649], [0.82740, 0.3925], [0.9697, 0.8192], [0.5979, 0.9392], [0.2184, 0.8191], [0.7148, 0.4351], [0.2395, 0.8646], [0.2867, 0.6768]]

idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


df_original = pd.DataFrame(cities,index=idx, columns=["x", "y"] )



iterations = 1000
temp=2
a = 0.99

# generate an initial random solution
random_start = df_original.sample(frac=1).reset_index()

sa = SA(random_start, iterations, temp, a)
scores, best_scores, temps, best_df = sa.run()




plt.scatter(df["x"], df["y"], label="original data", c="Red")
plt.plot(random_start["x"], random_start["y"],label="first random data", c="Blue" )
plt.plot(best_df["x"], best_df["y"],label="first random data", c="Green" )
plt.show()


