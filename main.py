import numpy as np


class k_means:
    def __init__(self, _data, _num_groups):
        self.data = _data
        centroids = np.zeros((_num_groups,) + (np.array(_data[0]).shape), dtype=float)
        self.centroids = centroids

    def Perform_Step(self):

        for value in self.data:

    def Calculate_Closest(centroids, data_point):
        distances = np.linalg.norm

        

k_means_set = k_means([[1,2],[5,2],[6,2],[8,6]], 3)

print(k_means_set.groups)
    