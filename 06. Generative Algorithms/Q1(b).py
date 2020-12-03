# Minimize the following function using the roulette wheel selection, with the mutation prob
# as 0.06 and mutating the single bit in the chromosome.
# (sinA  − cosB)^3
# where A and B are in the range [1,50).
# Also consider the crossover probability as 0.96 with
# a. single-point crossover
# b. two-point crossover
# Let the GA take the best solution with it as it evolves.
# Plot and print the evolution of GA throughout the generations.
# Print the best solution found by the GA

from geneticalgs import RealGA
import math
import matplotlib.pyplot as plt 

def fitness_function(list):
	return abs(math.pow((math.sin(list[0]) - math.cos(list[1])), 3))


# class geneticalgs.real_ga.RealGA( fitness_func = None, optim = 'max', selection = 'rank'
# 	mut_prob = 0.5, mut_type = 1, cross_prob = 0.95, cross_type = 1,
# 	elitism = True, tournament_size = None)

gen_alg = RealGA(fitness_function, optim = 'min', selection = 'roulette', 
			mut_prob = 0.06, mut_type = 1, cross_prob = 0.96, cross_type = 2)

gen_alg.init_random_population(20, 2, (1, 50))
gen_alg._adjust_to_interval(-1)
number_of_generations = 50
fitness_progress = gen_alg.run(number_of_generations)
print(gen_alg.best_solution)

x1 = list(range(len(fitness_progress)))

plt.plot(x1, fitness_progress, 'o')
plt.show()