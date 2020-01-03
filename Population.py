import random

def individual_in_population(indiv_1,population):
    for i in range(len(population)):
        if indiv_1 == population[i]:
            return True
    return False


def create_individual():
    # nb_layer,nb_percep,epoch,lr => [(1,5),(10,512),(21,301),(0.001,0.1)]
    nb_layer = int(random.randint(1,5))
    nb_percep = int(random.randint(10,512))
    epoch = int(random.randint(21,301))
    lr = round(random.uniform(0.001,0.1),3)
    indiv = [nb_layer,nb_percep,epoch,lr]
    print("[INDIVIDUAL CREATED]")
    return indiv

def create_population(number_of_individuals):
    
    population = [create_individual()] #first individual

    for i in range(number_of_individuals-1):
        indiv = create_individual()
        while individual_in_population(indiv,population) :
            indiv = create_individual()
        population.append(indiv)
    print("[ -- POPULATION CREATED]")
    return population

# print(create_population(10))