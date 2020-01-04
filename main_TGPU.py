#--- IMPORT DEPENDENCIES --------------------------------------------------------------+
import Data
import Differential_Evolution
from Differential_Evolution import Model
import sys,datetime
sys.stdout = open('TrainingsLog/train'+str(datetime.datetime.now().strftime("%Y%m%d%H%M"))+".log", 'w+')
import timeit

def gpu():
  with tf.device('/device:GPU:0'):
    #--- PREPARE DATA ---------------------------------------------------------------------+
    dataset = Data.preprocessing("dataset/haberman.data","dataset/test_01.data",0.8)

    #--- META-HEURISTIQUE -----------------------------------------------------------------+
    best_model = Differential_Evolution.minimize(dataset=dataset, popsize=10,maxiter=5)

    #--- TRAIN & SAVE THE BEST MODEL ------------------------------------------------------+
    Model.retrain_and_save(best_model,dataset,300)

print('YOUR PROGRAM START ON GPU:')
gpu_time = timeit.timeit('gpu()', number=1, setup="from __main__ import gpu")
print("---|[ GPU (s): ",datetime.timedelta(seconds=gpu_time)," |]---")
print("[H:M:S]",)
