import random

def shuffle(dataset):

    shuffledDS = dataset[:]
    for i in range(len(shuffledDS)):
        j = int(random.randint(0,(len(shuffledDS)-1))) #select a random element
        e = shuffledDS[i]
        shuffledDS[i] = shuffledDS[j]
        shuffledDS[j] = e

    return shuffledDS

# print(shuffle([[74,63,0],[75,62,1],[76,67,0],[77,65,3],[78,65,1],[83,58,2]]))

def data_from_file(path):

    finalDataset = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            ages = int(line[0:2])
            year_of_operation = int(line[3:5])
            if len(line) == 10:
                Nb_positive_axillary_nodes = int(line[6:8])
                if int(line[9:10]) == 1:
                    Survival = 0 
                else:
                    Survival = 1 
            else:
                Nb_positive_axillary_nodes = int(line[6:7])
                if int(line[8:9]) == 1:
                    Survival = 0
                else:
                    Survival = 1
            finalDataset.append([ages,year_of_operation,Nb_positive_axillary_nodes,Survival])
    print("[DATA_FROM_FILE]: Completed!")
    
    return finalDataset

def preprocessing(path_TrainData, path_TestData, train_perc=0.8):

    data = data_from_file(path_TrainData)
    t_data = data_from_file(path_TestData)

    size_data = len(data)
    size_train = int( size_data * train_perc )

    train_input = []
    train_target = []
    valid_input = []
    valid_target = []

    for i in range(size_train):
        train_input.append([ data[i][0],
                             data[i][1],
                             data[i][2] 
                            ])
        train_target.append(data[i][3])

    for i in range(size_train,size_data):
        valid_input.append([ data[i][0],
                             data[i][1],
                             data[i][2] 
                            ])
        valid_target.append(data[i][3])


    test_input = []
    test_target = []
    for i in range(len(t_data)):
        test_input.append([  data[i][0],
                             data[i][1],
                             data[i][2] 
                            ])
        test_target.append(t_data[i][3])
 
    return [train_input,train_target,valid_input,valid_target,test_input,test_target]

# x, y, r, e, z, d = preprocessing("dataset/haberman.data", "dataset/test_01.data", 0.8)
# print(x)