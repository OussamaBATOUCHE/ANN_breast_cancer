#--- IMPORT DEPENDENCIES --------------------------------------------------------------+
import Data
import Differential_Evolution
from Differential_Evolution import Model
import sys,datetime
sys.stdout = open('TrainingsLog/train'+str(datetime.datetime.now().strftime("%Y%m%d%H%M"))+".log", 'w+')

#--- PREPARE DATA ---------------------------------------------------------------------+
dataset = Data.preprocessing("dataset/haberman.data","dataset/test_01.data",0.8)

#--- META-HEURISTIQUE -----------------------------------------------------------------+
best_model = Differential_Evolution.minimize(dataset=dataset, popsize=5,maxiter=2)

#--- TRAIN & SAVE THE BEST MODEL ------------------------------------------------------+
Model.retrain_and_save(best_model,dataset,50)
# Model.retrain_and_save([2,100,100,0.001],dataset,10)
