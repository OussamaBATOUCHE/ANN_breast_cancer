from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
import pandas as pd

import batlib as bl
from batlib import tf

def preprocess_data(pathToTrainData,pathToTestData):
    data = bl.data_to_dictionary(pathToTrainData)
    t_data = bl.data_to_dictionary(pathToTestData)

    train_input = []
    train_target = []
    valid_input = []
    valid_target = []

    for i in range(244):
        train_input.append([data["ages"][i],data["year_of_operation"][i],data["nb_positive_axillary_nodes"][i]])
        train_target.append(data["survival"][i])

    for i in range(244,300):
        valid_input.append([data["ages"][i],data["year_of_operation"][i],data["nb_positive_axillary_nodes"][i]])
        valid_target.append(data["survival"][i])


    test_input = []
    test_target = []
    for i in range(6):
        test_input.append([t_data["ages"][i],t_data["year_of_operation"][i],t_data["nb_positive_axillary_nodes"][i]])
        test_target.append(t_data["survival"][i])
    print("ok")    
    return [train_input,train_target,valid_input,valid_target,test_input,test_target]

# x, y, r, e, z, d = preprocess_data("data/haberman.data","data/test_01.data")
# print(x)

def train(model_architecture,dataset):
    # Get data
    train_input, train_target, valid_input, valid_target, test_input, test_target = dataset

    # Create, compile, and train the model
    nb_layer = model_architecture[0]
    nb_perceptron = model_architecture[1]
    nb_epoch = model_architecture[2]
    l_rate = model_architecture[3]

    model = bl.create_model(nb_layer,nb_perceptron,['relu','sigmoid'])

    tensorboard = TensorBoard(log_dir="logs/")
    adam = tf.keras.optimizers.Adam(learning_rate=l_rate)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=["binary_accuracy"])

    # Train the model
    tensorboard = TensorBoard(log_dir="Logs/")
    model.fit(train_input,train_target,
            validation_data=(valid_input,valid_target),
            epochs=nb_epoch,
            callbacks=[tensorboard]
            )
    # print(model.summary())

    bin_loss, bin_accuracy = model.evaluate(test_input,test_target)
    print("TEST-Bin-Accuracy", bin_accuracy) 
    print("TEST-Bin-Loss", bin_loss)
    return bin_loss, bin_accuracy

# train([2,128,10,0.001],preprocess_data("data/haberman.data","data/test_01.data"))