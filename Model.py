import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import Cost
import Data
import datetime


def create( nb_layers, nb_percep, activationFunctions=["relu","sigmoid"]):
    """
     The fuction can be used to create models
     nb_layers can't be more than 5
     nb_percep is a list
     activationFunctions is a list
     feature_layers is a list
    """
    model = tf.keras.Sequential()
    # model.add(layers.Dense(128, input_dim=3,activation=activationFunctions[0]))
    for i in range(nb_layers):
        if i == 0:
            model.add(layers.Dense(nb_percep, activation=activationFunctions[0], input_dim=3))
        else:
            model.add(layers.Dense(nb_percep, activation=activationFunctions[0]))    

    model.add(layers.Dense(1, activation=activationFunctions[1]))
    return model

def train(model_architecture,dataset):
    # Get data
    train_input, train_target, valid_input, valid_target, test_input, test_target = dataset
    # Shuffle 
    train_input, train_target = Data.shuffle_ds(train_input, train_target)
    valid_input, valid_target = Data.shuffle_ds(valid_input, valid_target)

    # Create, compile, and train the model
    nb_layer = model_architecture[0]
    nb_perceptron = model_architecture[1]
    nb_epoch = model_architecture[2]
    l_rate = model_architecture[3]
    logfilename = str(nb_layer)+"_"+str(nb_perceptron)+"_"+str(nb_epoch)+"_"+str(l_rate)[2:]+"_"+str(datetime.datetime.now().strftime("%H%M%S"))

    model = create(nb_layer,nb_perceptron,['relu','sigmoid'])

    tensorboard = TensorBoard(log_dir="logs/"+logfilename+"/")
    adam = tf.keras.optimizers.Adam(learning_rate=l_rate)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=["binary_accuracy"])

    # Train the model
    model.fit(train_input,train_target,
            validation_data=(valid_input,valid_target),
            epochs=nb_epoch,
            callbacks=[tensorboard],
            verbose=1
            )
    # print(model.summary())

    bin_loss, bin_accuracy = model.evaluate(test_input,test_target)
    # print("TEST-Bin-Accuracy", bin_accuracy) 
    # print("TEST-Bin-Loss", bin_loss)
    return round(bin_loss,2), round(bin_accuracy,2), model

def retrain(model, dataset, iterations):  
    best_score = 2
    for i in range(iterations):
        metrics = train(model,dataset)
        score = Cost.score_loss_acc(metrics[0],metrics[1])
        if score < best_score:
            best_score = score

    return best_score

def retrain_and_save(model, dataset, iterations):  
    best_accuracy = 0
    score = 2
    for i in range(iterations):
        metrics_and_model = train(model,dataset)
        score_i = round(((1-metrics_and_model[1])+metrics_and_model[0]),2)
        print("--------------- END TRAIN with BIN-ACCURACY : ",metrics_and_model[1]," ----  LOSS : ",metrics_and_model[0]," ------ SCORE : ",score_i)
        if score_i < score:
            score = score_i
            best_accuracy = metrics_and_model[1]
            metrics_and_model[2].save("best_model.h5")
    print("++++++++++++  BEST MODEL SAVED ! With Binary Accuracy",best_accuracy," and score ",score)


