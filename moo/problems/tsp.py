from math import sqrt
from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial.distance import cdist


class TravellingSalesmanProblemV1:
    """The Traveling Salesman Problem (TSP) is a classic optimization
    problem in computer science and mathematics, where the goal is to
    find the shortest possible route for a salesman who must visit a
    given set of cities exactly once and return to the starting city.
    """

    def __init__(self) -> None:
        pass

    def evaluate_travelled_distance(self, cities: List[Tuple[int, int]]) -> float:
        "Using euclidean distance to calucatlate to total distance travelled"
        n_cities = len(cities)

        total_distance = 0.
        for i in range(n_cities):
            current_city = cities[i]
            if i+1 < n_cities:
                next_city = cities[i+1]
            else:
                next_city = cities[0]
            distance = sqrt((next_city[0]-current_city[0])**2 + (next_city[1]-current_city[1])**2)
            total_distance = total_distance + distance
        
        return total_distance


class TravellingSalesmanProblemV2:

    def __init__(self, cities: List[Tuple[int, int]]) -> None:
        self.cities = cities
        self.n_cities = len(cities)

    def evaluate_travelled_distance(self, travel_order: List[int]) -> float:
        total_distance = 0.
        for i in range(self.n_cities):
            
            current_city_index = travel_order[i]
            current_city = self.cities[current_city_index]

            if i+1 < self.n_cities:
                next_city_index = travel_order[i+1]
                next_city = self.cities[next_city_index]
            else:
                next_city_index = travel_order[0]
                next_city = self.cities[next_city_index]

            distance = sqrt((next_city[0]-current_city[0])**2 + (next_city[1]-current_city[1])**2)
            total_distance = total_distance + distance

        return total_distance


class TravellingSalesmanProblemV3:

    def __init__(self, cities: List[Tuple[int, int]]) -> None:
        self.cities = cities
        self.n_cities = len(cities)
        self.distance_matrix: Optional[np.ndarray] = None

        self._update_distance_matrix()

    def evaluate_travelled_distance(self, travel_order: List[int]) -> float:
        total_distance = 0.
        for i in range(self.n_cities):
            current_city_index = travel_order[i]
            if i+1 < self.n_cities:
                next_city_index = travel_order[i+1]
            else:
                next_city_index = travel_order[0]

            distance = self.distance_matrix[current_city_index, next_city_index]
            total_distance = total_distance + distance

        return total_distance

    def _update_distance_matrix(self) -> None:
        """Calculate and update the distance matrix."""
        matrix = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            current_city = self.cities[i]
            for j in range(self.n_cities):
                next_city = self.cities[j]

                distance = sqrt((next_city[0]-current_city[0])**2 + (next_city[1]-current_city[1])**2)
                matrix[i, j] = distance
        self.distance_matrix = matrix


class TravellingSalesmanProblemV4:

    def __init__(self, cities: List[Tuple[int, int]]) -> None:
        self.cities = cities
        self.n_cities = len(cities)
        self.distance_matrix: Optional[np.ndarray] = None

        self._update_distance_matrix()

    def evaluate_travelled_distance(self, travel_order: List[int]) -> float:
        total_distance = 0.
        current_travel_order = np.array(travel_order)
        next_travel_order = np.roll(current_travel_order, -1)
        for current_index, next_index in zip(current_travel_order, next_travel_order):
            distance = self.distance_matrix[current_index, next_index]
            total_distance = total_distance + distance
        return total_distance

    def _update_distance_matrix(self) -> None:
        self.distance_matrix = cdist(self.cities, self.cities, 'euclidean')
        

class TravellingSalesmanProblem:

    def __init__(self, cities: List[Tuple[int, int]]) -> None:
        self.cities: np.ndarray = np.array(cities)
        self.n_cities = len(cities)
        self.distance_matrix: Optional[np.ndarray] = None

        self._update_distance_matrix()


    def evaluate_travelled_distance(self, travel_order: List[int]) -> float:
        current_travel_order = np.array(travel_order) 
        next_travel_order = np.roll(current_travel_order, -1)

        distances = self.distance_matrix[current_travel_order, next_travel_order]
        return np.sum(distances)

    def _update_distance_matrix(self) -> None:
        self.distance_matrix = cdist(self.cities, self.cities, 'euclidean')


class TravellingSalesmanProblemShortForm:

    def __init__(self, cities: List[Tuple[int, int]]) -> None:
        self.cities = np.array(cities)
        self.n_cities = len(cities)
        self.distance_matrix = cdist(self.cities, self.cities, 'euclidean')

    def evaluate_travelled_distance(self, travel_order: List[int]) -> float:
        current_travel_order = np.array(travel_order) 
        next_travel_order = np.roll(current_travel_order, -1)
        return self.distance_matrix[current_travel_order, next_travel_order].sum()
    
