import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def distance(city1, city2):
    return np.linalg.norm(city1 - city2)
def ant_colony_optimization(cities, num_ants, num_iterations, alpha, beta, rho, q):
    num_cities = len(cities)
    pheromones = np.ones((num_cities, num_cities))  
    best_path = None
    best_path_length = np.inf
    for iteration in range(num_iterations):
        paths = []
        for ant in range(num_ants):
            path = [np.random.randint(num_cities)]
            visited = [False] * num_cities
            visited[path[0]] = True
            for _ in range(num_cities - 1):
                current_city = path[-1]
                probabilities = np.zeros(num_cities)
                for city in range(num_cities):
                    if not visited[city]:
                        probabilities[city] = (pheromones[current_city, city] ** alpha) * \
                                              ((1.0 / distance(cities[current_city], cities[city])) ** beta)
                probabilities = probabilities / np.sum(probabilities)
                next_city = np.random.choice(np.arange(num_cities), p=probabilities)
                path.append(next_city)
                visited[next_city] = True
            path.append(path[0])
            paths.append(path)
        pheromones *= (1 - rho)
        for path in paths:
            path_length = sum(distance(cities[path[i]], cities[path[i + 1]]) for i in range(num_cities))
            for i in range(num_cities):
                pheromones[path[i], path[i + 1]] += q / path_length
                pheromones[path[i + 1], path[i]] += q / path_length
            if path_length < best_path_length:
                best_path_length = path_length
                best_path = path[:-1]  
        yield cities[np.array(best_path)]
np.random.seed(0)
num_cities = 20
cities = np.random.rand(num_cities, 2)
num_ants = 10
num_iterations = 100
alpha = 1.0  
beta = 5.0  
rho = 0.5  
q = 100.0 
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.scatter(cities[:, 0], cities[:, 1], marker='o', c='red', s=50)
def update(frame):
    line.set_data(frame[:, 0], frame[:, 1])
    return line,
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Ant Colony Optimization for TSP')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ani = animation.FuncAnimation(fig, update, frames=ant_colony_optimization(cities, num_ants, num_iterations, alpha, beta, rho, q),
                              interval=200, blit=True)
plt.show()
