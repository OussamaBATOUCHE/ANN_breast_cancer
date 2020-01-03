def score_loss_acc(loss,accuracy):
    # loss = round(loss,1)
    # los_score = max(0,round((0.5 - loss),1)) * 10
    # acc_score = max(0,(accuracy - 0.70)) * 10
    return round(((1-accuracy)+loss),2)