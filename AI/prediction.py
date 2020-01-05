import tensorflow as tf
from tensorflow import keras

print("PREDICTION ------- ")
# using a trainned model
test_input = [[74,63,0],[75,62,1],[76,67,0],[77,65,3],[78,65,1],[83,58,2]]
test_target = [0,0,0,0,1,1]
trained_model = keras.models.load_model("best_model.h5")
test_loss , test_acc = trained_model.evaluate(test_input,test_target)
print("FROM TRAINED MODEL")
print("tested acc : ",test_acc)
print("tested loss : ",test_loss)

prediction = trained_model.predict(test_input)
print(prediction)