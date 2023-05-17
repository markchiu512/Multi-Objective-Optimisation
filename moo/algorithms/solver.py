from typing import TYPE_CHECKING, Dict
from itertools import permutations

import numpy as np

if TYPE_CHECKING:
    from moo.problems.tsp import TravellingSalesmanProblem


class DummyAlgorithm:
    """A dummy algorithm as an example, which does not optimisation anything
    but return list of range from 0 to n_cities only.
    """
    
    def __init__(self, tsp: 'TravellingSalesmanProblem') -> None:
        self.problem = tsp

    def solve(self) -> Dict:
        variables = list(range(self.problem.n_cities))
        objective = self.problem.evaluate_travelled_distance(variables)
        output_dict = {
            'variables': variables,
            'objective': objective
        }
        return output_dict


class DummyAlgorithmAlterned:
    """A dummy algorithm as an example, which does not optimisation anything
    but return list of range from 0 to n_cities only.
    """

    def solve(self, tsp: 'TravellingSalesmanProblem') -> Dict:
        variables = list(range(tsp.n_cities))
        objective = tsp.evaluate_travelled_distance(variables)
        output_dict = {
            'variables': variables,
            'objective': objective
        }
        return output_dict


class GreedyAlgorithmV1:
    """A greedy strategy for the travelling salesman problem (which is of high 
    computational complexity) is the following heuristic: "At each step of the
    journey, visit the nearest unvisited city." This heuristic does not intend
    to find the best solution, but it terminates in a reasonable number of steps; 
    finding an optimal solution to such a complex problem typically requires 
    unreasonably many steps. """

    def solve(self, tsp: 'TravellingSalesmanProblem' ) -> Dict:
        distance_matrix = tsp.distance_matrix
         #print(distance_matrix)

        # Before salesman travels
        unvisited_cities = list(range(tsp.n_cities))
        travel_order = []
        
        # We set the Saleman starting point
        starting_point = 0
        travel_order.append(starting_point) # [0]
        unvisited_cities.remove(starting_point) # [1,2,3,4,5,6,7]

        # Starting optimising
        while len(travel_order) < tsp.n_cities:
            current_city = travel_order[-1] # 2
            # including itself and visited cities
            current_city_distances_to_all_cities = distance_matrix[current_city, :] # [1, 4, 0, 4, 4, 6, 3, 1]
            current_city_distances_to_unvisited_cities = current_city_distances_to_all_cities[unvisited_cities] # [4, 4, 4, 6, 3, 1]]

            next_city_index = np.argmin(current_city_distances_to_unvisited_cities) # 5
            next_city = unvisited_cities[next_city_index] # 6

            travel_order.append(next_city) # [0,2,7,6]
            unvisited_cities.remove(next_city) # [1,3,4,5]

        objective = tsp.evaluate_travelled_distance(travel_order)
        output_dict = {
            'variables': travel_order,
            'objective': objective
        }
        return output_dict

class GreedyAlgorithmV2:

    def solve(self, tsp: 'TravellingSalesmanProblem') -> Dict:
        distance_matrix = tsp.distance_matrix

        unvisited_cities = np.arange(tsp.n_cities, dtype=np.int16) #[0, 1, 2, 3]
        unvisited_cities_mask = np.ones(tsp.n_cities, dtype=np.bool_) #[T, T, T, T]
        travel_order = np.zeros(tsp.n_cities, dtype=np.int16) #(0, 0, 0, 0)

        for i in range(tsp.n_cities-1): 
            current_city = travel_order[i] #0 #2
            unvisited_cities_mask[current_city]=False #[F, T, T, T] #[F, T, F, T]
            current_city_distances_to_all_cities = distance_matrix[current_city, :] #[0, 96, 4.2, 8] #[9, 2, 0, 3]
            current_city_distances_to_unvisited_cities = current_city_distances_to_all_cities[unvisited_cities_mask] #[96, 4.2, 8] #[2, 3]

            next_city_index = np.argmin(current_city_distances_to_unvisited_cities) #1 #0
            next_city = unvisited_cities[unvisited_cities_mask][next_city_index] # [1,2,3] [1] = [2] # [1,3] [0] = [1]

            travel_order[i+1] = next_city #[0, 2, 0, 0] #[0, 2, 1, 0]

        objective = tsp.evaluate_travelled_distance(travel_order)
        output_dict = {
            'variables': travel_order.tolist(),
            'objective': objective
        }
        return output_dict
    

class BruteForceV1:

    def solve(self, tsp: 'TravellingSalesmanProblem'):
        distance_matrix = tsp.distance_matrix
        n_cities = tsp.n_cities

        city_permutations = permutations(range(1, n_cities))

        min_distance = float('inf')
        travel_order = None

        # Calculate the total distance for each permutation
        for perm in city_permutations:
            route = [0] + list(perm) + [0]  # Add the starting city to the start and end of the route
            distance = sum(distance_matrix[route[i-1]][route[i]] for i in range(1, len(route)))

            # Update the minimum distance and best route if this route is better than the current best
            if distance < min_distance:
                min_distance = distance
                travel_order = route
        
        output_dict = {
            'variables': travel_order,
            'objective': min_distance
        }
        return output_dict
    
