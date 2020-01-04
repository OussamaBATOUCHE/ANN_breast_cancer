#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import Population as pl
from Population import random
import Model
import Cost

#--- FUNCTIONS ----------------------------------------------------------------+


def ensure_bounds(vec, bounds):
    # nb_layer,nb_percep,epoch,lr => [(1,5),(10,512),(21,301),(0.001,0.1)]

    vec_new = []
    # cycle through each variable in vector 
    for i in range(len(vec)):

        # variable exceedes the minimum boundary
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])

        # variable exceedes the maximum boundary
        if vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])

        # the variable is fine
        if bounds[i][0] <= vec[i] <= bounds[i][1]:
            vec_new.append(vec[i])
        
    return vec_new


#--- MAIN ---------------------------------------------------------------------+

def minimize(dataset, cost_func=Cost.score_loss_acc, bounds=[(1,5),(10,512),(21,301),(0.001,0.1)], popsize=5, mutate=0.5, recombination=0.7, maxiter=20):

    #--- INITIALIZE A POPULATION (step #1) ----------------+
    
    population = pl.create_population(popsize)
            
    #--- SOLVE --------------------------------------------+
    # save generations best solution
    all_gen_best_score = []
    all_gen_best = []
    # cycle through each generation (step #2)
    for i in range(1,maxiter+1):
        print ('GENERATION:',i)

        gen_scores = [] # score keeping

        # cycle through each individual in the population
        for j in range(0, popsize):

            #--- MUTATION (step #3.A) ---------------------+
            
            # select three random vector index positions [0, popsize), not including current vector (j)
            canidates = list(range(0,popsize))
            canidates.remove(j)
            random_index = random.sample(canidates, 3)

            model_1 = population[random_index[0]]
            model_2 = population[random_index[1]]
            model_3 = population[random_index[2]]
            model_target = population[j]         # target individual

            # subtract model_3 from model_2, and create a new vector (model_diff)
            model_diff = [model_2_i - model_3_i for model_2_i, model_3_i in zip(model_2, model_3)]

            # multiply model_diff by the mutation factor (F) and add to model_1
            v_donor = [model_1_i + mutate * model_diff_i for model_1_i, model_diff_i in zip(model_1, model_diff)]
            v_donor = ensure_bounds(v_donor, bounds)

            #--- RECOMBINATION (step #3.B) ----------------+

            v_trial = []
            for k in range(len(model_target)):
                crossover = random.random()
                if crossover <= recombination:
                    v_trial.append(v_donor[k])

                else:
                    v_trial.append(model_target[k])
            v_trial[0] = int(v_trial[0] )
            v_trial[1] = int(v_trial[1] )
            v_trial[2] = int(v_trial[2] )
            
            print ('Trail : ',v_trial)    
            print ('Target : ',model_target)    
            #--- GREEDY SELECTION (step #3.C) -------------+
            # Train each model 5 times and take the best
            score_trial = Model.retrain(v_trial,dataset,5)
            score_target = Model.retrain(model_target,dataset,5)
                
            # Select the best
            # score_trial  = cost_func(v_trial)
            # score_target = cost_func(model_target)
            print('score_trail : ',score_trial)
            print('score_target : ',score_target)
            if score_trial < score_target: # minimize score
                population[j] = v_trial
                print('Population id = ',j,'changée a : ',v_trial)
                gen_scores.append(score_trial)
                print ('   >',score_trial,' vec : ',v_trial)

            else:
                print ('   >',score_target,' target : ',model_target)
                gen_scores.append(score_target)

        #--- SCORE KEEPING --------------------------------+

        gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
        gen_best = min(gen_scores)                                  # fitness of best individual
        gen_sol = population[gen_scores.index(gen_best)]     # solution of best individual

        print ('      > GENERATION AVERAGE:',gen_avg)
        print ('      > GENERATION BEST:',gen_best)
        print ('         > BEST SOLUTION:',gen_sol,'\n')

        print("--------- DETIALS ++++++++++++++++++++ ")
        print('> GENERATION ',i,' ALL SOLUTIONS [ SCORE : [MODEL] ]')
        for s in range(len(gen_scores)):
            print('[ ',gen_scores[s],' : ',population[s],' ]')
        
        all_gen_best.append(gen_sol) #add the best soluction in this generation
        all_gen_best_score.append(gen_best) #add the best score in this generation

    #--- Select the best generation and best solution -----+
    # All generations best models and there scores
        print(" ------------[ D.E MINIMIZER FINISHED ")
        print("    > DETAILS ")
        for k in range(len(all_gen_best)):
            print('       - [ ',all_gen_best_score[k],' : ',all_gen_best[k],' ]')

    # Selection the best of the best        
    best_gens_sol = all_gen_best[all_gen_best_score.index(min(all_gen_best_score))]
    print("      > RESULT ")
    print("-------------> The best model in all generations : ",best_gens_sol,'\n   -- SCORE : ',min(all_gen_best_score))
    return best_gens_sol

#--- CONSTANTS ----------------------------------------------------------------+

#cost_func = cost                                         # Cost function
#bounds = [(1,5),(10,512),(21,301),(0.001,0.1)]            # Bounds [(nblayer_min, nblayer_max), (nbperc_min, nbperc_max),(epoch_min, epoch_max),(lr_min, lr_max)]
#popsize = 10                                              # Population size, must be >= 4
#mutate = 0.5                                              # Mutation factor [0,2]
#recombination = 0.7                                       # Recombination rate [0,1]
#maxiter = 20                                              # Max number of generations (maxiter)

#--- RUN ----------------------------------------------------------------------+

# main(cost_func, bounds, popsize, mutate, recombination, maxiter)

#--- END ----------------------------------------------------------------------+