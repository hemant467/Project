import random
import numpy as np

SBOX_SIZE = 256
POPULATION_SIZE = 100
GENERATIONS = 100
INITIAL_MUTATION_RATE = 0.05
MUTATION_RATE_ADAPTATION_FACTOR = 0.98
ELITISM_PERCENTAGE = 0.1

def evaluate_sbox(s_box):
    balance_score = abs(sum(s_box) - (SBOX_SIZE * (SBOX_SIZE - 1) / 2))
    return balance_score

def genetic_algorithm():
    population = [np.random.permutation(range(SBOX_SIZE)) for _ in range(POPULATION_SIZE)]
    mutation_rate = INITIAL_MUTATION_RATE

    for generation in range(GENERATIONS):
        fitness_scores = [evaluate_sbox(s_box) for s_box in population]

        elite_count = int(ELITISM_PERCENTAGE * POPULATION_SIZE)
        elite_indices = np.argsort(fitness_scores)[:elite_count]
        elite_population = [population[i] for i in elite_indices]

        new_population = elite_population
        for _ in range((POPULATION_SIZE - elite_count) // 2):
            parent1, parent2 = random.choices(elite_population, k=2)
            crossover_point = random.randint(0, SBOX_SIZE - 1)
            child = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
            if random.random() < mutation_rate:
                mutation_point = random.randint(0, SBOX_SIZE - 1)
                child[mutation_point] = random.randint(0, SBOX_SIZE - 1)
            new_population.append(child)

        population = new_population
        mutation_rate *= MUTATION_RATE_ADAPTATION_FACTOR

    best_s_box = min(population, key=evaluate_sbox)
    return best_s_box

def analyze_sbox(s_box):
    s_box_matrix = np.reshape(s_box, (16, 16))
    print("S-Box Output:")
    for row in s_box_matrix:
        print(" ".join([f"{val:3}" for val in row]))

best_s_box = genetic_algorithm()
analyze_sbox(best_s_box)

