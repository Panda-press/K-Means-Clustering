import numpy as np
import random as rnd


class k_means:
    def __init__(self, _data, _num_groups):
        self.data = _data
        centroids = np.zeros((_num_groups,) + (np.array(_data[0]).shape), dtype=float)
        self.centroids = centroids

    def Randomize_Centroids(self):
        s = np.random.default_rng().normal(0, np.std(self.data), 1000)
        for i in range(0, self.centroids.shape[0]):
            for k in range(0, self.centroids.shape[1]):
                self.centroids[i,k] = s[rnd.randint(0, 1000)]

    def Calculate_Closest(centroids, data_point):
        distances = np.array([])
        for center in centroids:
            distances = np.append(distances, np.linalg.norm(center - data_point))

        #print(distances)
        return np.argmin(distances)

    def Calculate_New_Centroids(centroids, data_points):
        totals = np.zeros_like(centroids)
        tally = np.zeros(centroids.shape[0])
        for point in data_points:
            closest = k_means.Calculate_Closest(centroids, point)
            totals[closest] += point
            tally[closest] += 1

        new_centroids = np.zeros_like(centroids)
        for i in range(0, totals.shape[0]):
            new = np.array([])
            for k in range(0, totals.shape[1]):
                new = np.append(new, [totals[i,k] / tally[i]])

            new_centroids[i] = new

        return new_centroids
        


        

k_means_set = k_means([[1,2],[-5,2],[-6,-2],[8,-6]], 2)
k_means_set.Randomize_Centroids()

print(k_means_set.centroids)
print(k_means.Calculate_New_Centroids(k_means_set.centroids, k_means_set.data))

#print(k_means.Calculate_Closest(np.array([[1,2],[5,2],[6,2],[8,6]]), np.array([6,2])))
    