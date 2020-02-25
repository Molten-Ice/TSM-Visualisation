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

no_of_generations = 1000  # number of rounds of breeding
pop_size = 120  # each generation has this number of routes
how_many_to_kill = 75  # number of routes to remove from population each round
prob_mut = 0.5  # probability that mutation occurs on a route
percentage_mut = 0.1
percentage_kill = 0.8

# def create_new_route():
#     start = np.array([0])
#
#     intermediate_steps = np.random.permutation(np.arange(1, N - 1))
#
#     end = np.array([N - 1])
#
#     temp = np.append(start, intermediate_steps)
#     route = np.append(temp, end)
#
#     return route


def create_new_route(num_places):
    """
    :param num_places: number of places to permute
    :return: route: permutation of places
    """

    start = np.array([0])  # start of route

    intermediate_steps = np.random.permutation(np.arange(1, num_places - 1))  # places in between

    end = np.array([num_places - 1])  # end of route

    # put route together
    temp = np.append(start, intermediate_steps)
    route = np.append(temp, end)

    return route


def crossover(a, b):
    """
    :param a: route to crossover
    :param b: route to crossover
    :return: a, b: routes after crossover
    """

    a_copy = a.copy()
    b_copy = b.copy()

    intersection = -1  # first intersection point index

    for i in range(1, len(a_copy) - 2):  # find the first intersection point
        if a_copy[i] == b_copy[i]:
            intersection = i
            break

    if intersection >= 0:  # if intersection is not empty

        random_index = intersection

        while random_index == intersection:  # generate a random index that is not intersection
            random_index = random.randint(1, len(a) - 2)

        # swap intersection place with random place
        temp = a_copy[intersection]
        a_copy[intersection] = a_copy[random_index]
        a_copy[random_index] = temp

        # this code is unreachable
        # while random_index == intersection:  # generate a random index that is not intersection
        #    random_index = random.randint(1, len(a) - 2)

        # swap intersection place with random place
        temp = b_copy[intersection]
        b_copy[intersection] = b_copy[random_index]
        b_copy[random_index] = temp

    return a_copy, b_copy


def mutate(routes, our_map):
    """
    :param routes: routes to mutate (or not)
    :param prob_mut: probability of performing mutation
    :return: routes after mutation
    """


    ranked_pop = [routes[i] for i in sort_population(routes, our_map)]
    new_routes = ranked_pop[:-(int(len(routes) * percentage_mut))]
    worse_routes = ranked_pop[-(int(len(routes) * percentage_mut)):]
    for a in worse_routes:
        if random.random() < prob_mut:  # probability of running following code is prob_mut
            #print("Probably never running this. Probably.")
            i = random.randint(1, len(a) - 2)  # random place position
            j = i

            while j == i:  # make sure j is not equal to i
                j = random.randint(1, len(a) - 2)

            # swap two places
            temp = a[i]
            a[i] = a[j]
            a[j] = temp

        np.append(new_routes, a)
    return new_routes


def fitness(a, our_map):
    """
    Determines fitness score of route.
    Score is defined by the total distance.
    :param a: route which we are finding fitness of
    :param our_map: adjacency matrix of places
    :return: fitness score
    """

    score = 0

    for i in range(0, len(a) - 1):
        score += our_map[a[i]][a[i + 1]]  # distance between successive places

    return score


def create_generation(pop_size, our_map, num_places):
    population = []

    for i in range(0, pop_size):
        population.append(create_new_route(num_places))

    return population


def score_population(population, our_map):
    scores = []

    for i in range(0, len(population)):
        scores += [fitness(population[i], our_map)]

    return scores


def sort_population(population, our_map):
    scores = score_population(population, our_map)
    np_scores = np.array(scores)
    return np_scores.argsort().tolist()


def best_in_population(population, our_map):
    best = sort_population(population, our_map)[0]
    best_route = population[best]

    return best_route


def fitness_of_best_in_population(population, our_map):
    fit = fitness(best_in_population(population, our_map), our_map)

    return fit


def selection(population, our_map):
    ranked_pop = sort_population(population, our_map)
    survival_of_the_fittest = ranked_pop[: (int(len(population)*(1-percentage_kill)))]
    return [population[i] for i in np.array(survival_of_the_fittest)]


def breeding(population, our_map, num_places):
    children = []
    keep = 4
    # for i in range(0, keep):
    #     children.append(population[fittest[i]])
    while len(children) < pop_size:
        parent_1_index = random.randint(0, len(population) - 1)
        parent_2_index = parent_1_index
        while parent_1_index == parent_2_index:
            parent_2_index = random.randint(0, len(population) - 1)
        child = crossover(population[parent_1_index], population[parent_2_index])[0]
        # child = population[fittest[parent_1_index]]
        children.append(child)

    # for i in range(keep, len(fittest) - 1, 2):
    #     child_1 = crossover(population[fittest[i]], population[fittest[i + 1]])[0]
    #     child_2 = crossover(population[fittest[i]], population[fittest[i + 1]])[1]
    #     children.append(child_1)
    #     children.append(child_2)
    # while len(children) < pop_size:
    #     counter += 1
    #     new_route = create_new_route(num_places)
    #     children.append(new_route)

    return np.array(children)


def main():
    # our_map = initialise_map(lower_limit)
    names_of_locations_temp = ["kelseys", "the fat pug", "the town house", "the old library", "the clarendon",
                               "the benjamin satchwell", "murphy's bar", "The Royal Pug", "the white house",
                               "The Drawing Board"]
    names_of_locations = [name.lower() for name in sorted(names_of_locations_temp)]
    generator = MapGenerator(names_of_locations)
    locations = generator.decodeLocations()
    N = len(names_of_locations)

    num_to_object = {}
    object_to_num = {}
    for counter, location_object in enumerate(locations):
        num_to_object[counter] = location_object
        object_to_num[location_object] = counter
    our_map = generator.adjacency_matrix_generator()

    population = create_generation(pop_size, our_map, N)

    best_routes = []
    best_scores = []

    for i in range(0, no_of_generations):
        # Recording test routes
        current_best = best_in_population(population, our_map)
        best_routes.append(current_best)
        best_scores.append(fitness(current_best, our_map))
        
        population = selection(population, our_map)
        population = breeding(population, our_map, N)
        population = mutate(population, our_map)


    last = None
    for current_route in best_routes[-10:]:
        locations_to_render = [num_to_object[x] for x in current_route]
        if last != locations_to_render:
            generator.renderLocations(locations_to_render)
        last = locations_to_render

    plt.plot(np.arange(0, no_of_generations), best_scores)
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
