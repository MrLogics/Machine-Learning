from geneticalgs import RealGA
import math
import matplotlib.pyplot as plt 

def fitness_function(x):
	return abs(-math.pow(x,3) - math.pow(x,2) + x + 12)


# class geneticalgs.real_ga.RealGA( fitness_func = None, optim = 'max', selection = 'rank'
# 	mut_prob = 0.5, mut_type = 1, cross_prob = 0.95, cross_type = 1,
# 	elitism = True, tournament_size = None)

gen_alg = RealGA(fitness_function, optim = 'max')

gen_alg.init_random_population(20, 1, (0, 1000))
gen_alg._adjust_to_interval(-1)
number_of_generations = 50
fitness_progress = gen_alg.run(number_of_generations)
print(gen_alg.best_solution)

x1 = list(range(len(fitness_progress)))

plt.plot(x1, fitness_progress, 'o')
plt.show()