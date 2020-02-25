import numpy as np
import random
from random import randint
import copy
import matplotlib.pyplot as plt
import seaborn as sb
from map import *
import time

# ==============================
# Define constants
# ==============================

no_of_generations = 100  # number of rounds of breeding
N = 10
pop_size = 100  # each generation has this number of routes
how_many_to_kill = 75  # number of routes to remove from population each round
prob_mut = 0.0001  # probability that mutation occurs on a route


def create_new_route():
    start = np.array([0])

    intermediate_steps = np.random.permutation(np.arange(1, N - 1))

    end = np.array([N - 1])

    temp = np.append(start, intermediate_steps)
    route = np.append(temp, end)

    return route


def crossover(a, b):
    c = []

    for i in range(1, N - 2):
        if (a[i] == b[i]):
            c.append(i)

    if c != []:

        index = c[0]

        while (index == c[0]):
            index = random.randint(1, N - 2)

        temp = a[c[0]]
        a[c[0]] = a[index]
        a[index] = temp

        while (index == c[0]):
            index = random.randint(1, N - 2)

        temp = b[c[0]]
        b[c[0]] = b[index]
        b[index] = temp

    return a, b


def mutate(a, prob_mut):
    if random.random() > prob_mut:
        i = random.randint(1, N - 2)
        j = random.randint(1, N - 2)

        temp = a[i]
        a[i] = a[j]
        a[j] = temp

    return a


def fitness(a, our_map):
    score = 0

    for i in range(0, N - 1):
        score += our_map[a[i]][a[i + 1]]

    return score


def create_generation(pop_size, our_map):
    population = []

    for i in range(0, pop_size):
        population.append(create_new_route())

    return population


def score_population(population, our_map):
    scores = []

    for i in range(0, len(population)):
        scores += [fitness(population[i], our_map)]

    return scores


def sort_population(population, our_map):
    scores = score_population(population, our_map)

    np_scores = np.array(scores)

    return np_scores.argsort()


def best_in_population(population, our_map):
    best = sort_population(population, our_map)[0]
    best_route = population[best]

    return best_route


def fitness_of_best_in_population(population, our_map):
    fit = fitness(best_in_population(population, our_map), our_map)

    return fit


def remove_from_pop(how_many_to_kill, population, our_map):
    ranked_pop = sort_population(population, our_map)
    survival_of_the_fittest = ranked_pop[: (len(population) - how_many_to_kill)]

    return np.array(survival_of_the_fittest)


def breeding(population, our_map):
    fittest = remove_from_pop(how_many_to_kill, population, our_map)
    children = []

    keep = 4

    for i in range(0, 4):
        children.append(population[fittest[i]])

    for i in range(4, len(fittest) - 1, 2):
        child_1 = crossover(population[fittest[i]], population[fittest[i + 1]])[0]
        child_2 = crossover(population[fittest[i]], population[fittest[i + 1]])[1]
        child_1 = mutate(child_1, prob_mut)
        child_2 = mutate(child_2, prob_mut)
        children.append(child_1)
        children.append(child_2)

    while (len(children) < pop_size):
        new_route = create_new_route()
        children.append(new_route)

    return np.array(children)


def main():
    # our_map = initialise_map(lower_limit)
    names_of_locations_temp = ["kelseys", "the fat pug", "the town house", "the old library", "the clarendon",
                               "the benjamin satchwell", "murphy's bar", "The Royal Pug", "the white house",
                               "The Drawing Board"]
    names_of_locations = [name.lower() for name in sorted(names_of_locations_temp)]
    generator = MapGenerator(names_of_locations)
    locations = generator.decodeLocations()

    num_to_object = {}
    object_to_num = {}
    for counter, location_object in enumerate(locations):
        num_to_object[counter] = location_object
        object_to_num[location_object] = counter

    our_map = generator.adjacency_matrix_generator()
    population = create_generation(pop_size, our_map)

    route = []
    fitnesses = []

    for i in range(0, no_of_generations):
        route.append(best_in_population(population, our_map))
        fitnesses.append(fitness_of_best_in_population(population, our_map))
        breeding(population, our_map)

        population = breeding(population, our_map)

    last = None
    for current_route in route:
        locations_to_render = [num_to_object[x] for x in current_route]
        if last != locations_to_render:
            generator.renderLocations(locations_to_render)
        last = locations_to_render

    plt.plot(np.arange(0, no_of_generations), fitnesses)
    plt.ylabel('fitness')
    plt.xlabel('no. of generations')
    plt.show()


main()

"""
What we want it in this kinda form:
population = fitness(networks)
population = selection(networks)
population = crossover(networks)
population = mutate(networks)
Look at https://github.com/Molten-Ice/AI/blob/master/Hyperparameter%20optimisation%20using%20a%20Genetic%20algorithm 
to get an idea of code style
#
"""
