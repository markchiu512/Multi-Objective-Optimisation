from typing import TYPE_CHECKING, Dict

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
        # print(distance_matrix)

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


