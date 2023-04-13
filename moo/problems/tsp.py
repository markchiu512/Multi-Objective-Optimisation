from math import sqrt
from typing import List, Tuple

class TravellingSalesmanProblem:
    """The Traveling Salesman Problem (TSP) is a classic optimization
    problem in computer science and mathematics, where the goal is to
    find the shortest possible route for a salesman who must visit a
    given set of cities exactly once and return to the starting city.
    """

    def __init__(self) -> None:
        pass

    def evaluate_travelled_distance(self, cities: List[Tuple[int]]) -> float:
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

if __name__ == "__main__":
    tsp = TravellingSalesmanProblem()
    input_cities = [(0,1), (1,2), (2,3), (4,0), (3,2), (1,-6)]
    total_distance = tsp.evaluate_travelled_distance(cities=input_cities)
    print(total_distance)