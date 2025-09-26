import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the excel file and make sure it's read correctly
# file = pd.read_excel("Mortality_incidence_sociodemographic_and_clinical_data_in_COVID19_patients.xlsx")
# print(file.head())
# print("-"*30)
# print(file.info())

# Convert the excel file to a csv file and make sure it's read correctly
# file.to_csv("Mortality_incidence_sociodemographic_and_clinical_data_in_COVID19_patients.csv")
db = pd.read_csv("Mortality_incidence_sociodemographic_and_clinical_data_in_COVID19_patients.csv")
# print(db.head())
# print("-"*30)
# print(db.info())
# print("-"*30)
# print(db.shape)
# print("-"*30)

# Drop the rows with missing values
new_db = db.dropna()


# Effective features: Age.1, MI, CHF, CVD, DM Simple, DM Complicated, COPD, Renal Disease, DEMENT, Stroke, Seizure, OldOtherNeuro

# new_db["Age.1"] = new_db["Age.1"]/100  # Normalise age by dividing by 100

x_train = np.array(new_db[["Age.1", "MI", "CHF", "CVD", "DM Simple", "DM Complicated","COPD", "Renal Disease", "DEMENT", "Stroke", "Seizure", "OldOtherNeuro"]])

y_train = np.array(new_db["Death"])
###############
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
###############

w_init = np.zeros(x_train[0].shape)
b_init = 0


# sigmoid function
def sigmoid(x, w, b):
    return 1/(1 + np.exp(-(np.dot(w, x)+ b)))


def cost_function(x,y,w,b):
    cost = 0 
    for i in range(len(x)):
        # print(f"Index: {i}")
        z = sigmoid(x[i], w, b)
        # print(f"sigmoid: {z}")
        cost += -y[i]*np.log(z) - (1-y[i])*np.log(1-z)
        # print(f"Cost: {cost}")
        # print(f"final cost: {cost/len(x)}")
    return cost/len(x)


# derivative of cost function with respect to b
def dj_db(x, y, w, b):
    dj_db = 0
    for i in range(len(x)):
        err = sigmoid(x[i], w, b) - y[i]
        dj_db += err
    dj_db = dj_db/len(x)
    return dj_db


# derivative of cost function with respect to w 
def dj_dw(x, y, w, b):
    dj_dw = np.zeros(w.shape)
    for i in range(len(x)):
        err = sigmoid(x[i], w, b) - y[i]
        dj_dw += err * x[i]
    dj_dw = dj_dw/len(x)
    return dj_dw


def gradient_descent(x, y, w, b, learning_rate):
    cost,new_cost = 10000,1000 
    while (cost-new_cost) > 1e-6: 
        cost = cost_function(x,y,w,b)
        w = w - learning_rate*dj_dw(x, y, w, b)
        b = b - learning_rate*dj_db(x, y, w, b)
        new_cost = cost_function(x,y,w,b)
        print(f"Cost: {new_cost}")
    return w, b

# After testing multiple learning rates, 0.0001 seems to be the best
# w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, 0.001)

# print(f"w_final: {w_final}")
# print(f"b_final: {b_final}")


## The result of w_final and b_final after running gradient descent with normalising age are stored to speed up the run time of the program:
w_final = np.array([0.60798061,  0.01324616,  0.02119495, -0.0106811,  -0.02400938, -0.04393118, 0.03610855,  0.07490349,  0.01373136,  0.11121306,  0.01865869,  0.03534338])
b_final = -1.0959193809146979

## The result of w_final and b_final after running gradient descent without normalising age:
# w_final = np.array([-1.42845494e-02, -1.64697042e-05, -3.09810982e-05, -3.71334511e-05, -6.96377584e-05, -5.86839402e-05, -5.59714215e-06, -438724930e-05, -1.99598788e-05, 1.25707455e-05, -3.87570952e-06, -2.61773336e-06])
# b_final = -0.00048313246152727435



def calculate_probability(w,b,x):
    prob = sigmoid(x, w, b)
    return prob

# example usage:
# print(f"probability:{calculate_probability(w_final, b_final, x_train[0])}")

def choose_boundary(w, b, x, y):
    best_boundary = 0
    best_correct = 0
    boundaries = np.arange(0.0, 1.01, 0.01)
    for boundary in boundaries:
        correct = 0
        for i in range(len(x)):
            prob = calculate_probability(w, b, x[i])
            predict = 1 if prob >= boundary else 0
            if predict == y[i]:
                correct += 1
        if correct > best_correct:
            best_correct = correct
            best_boundary = boundary
    # print(f"Best correct: {best_correct} out of {len(x)}")
    # print(f"Accuracy: {best_correct/len(x)*100}%")
    return best_boundary

# the best boundary for the current model is 0.54
best_boundary = choose_boundary(w_final,b_final,x_train,y_train)


def prediction(w, b, boundary, age, mi, chf, cvd, dm_simple, dm_complicated, copd, renal_disease, dement, stroke, seizure, old_other_neuro):
    odds = calculate_probability(w, b, np.array([age, mi, chf, cvd, dm_simple, dm_complicated, copd, renal_disease, dement, stroke, seizure, old_other_neuro]))
    if odds >= boundary: 
        return 1
    else:
        return 0
    
# # example usage:
# print(prediction(w_final, b_final, best_boundary, x_train[1][0], x_train[1][1], x_train[1][2], x_train[1][3], x_train[1][4], x_train[1][5], x_train[1][6], x_train[1][7], x_train[1][8], x_train[1][9], x_train[1][10], x_train[1][11]))

# print(prediction(w_final, b_final, best_boundary, x_train[2][0], x_train[2][1], x_train[2][2], x_train[2][3], x_train[2][4], x_train[2][5], x_train[2][6], x_train[2][7], x_train[2][8], x_train[2][9], x_train[2][10], x_train[2][11]))

# print(prediction(w_final, b_final, best_boundary, x_train[3][0], x_train[3][1], x_train[3][2], x_train[3][3], x_train[3][4], x_train[3][5], x_train[3][6], x_train[3][7], x_train[3][8], x_train[3][9], x_train[3][10], x_train[3][11]))


# tp = true positive, tn = true negative, fn = false negative, fp = false positive 
def confusion_matrix(w, b, x, y, boundary = best_boundary):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(x)):
        pred = prediction(w,b,boundary,x[i][0], x[i][1], x[i][2], x[i][3], x[i][4], x[i][5], x[i][6], x[i][7], x[i][8], x[i][9], x[i][10], x[i][11])
        if pred == 1 and y[i] == 1:
            tp += 1
        elif pred == 0 and y[i] == 0: 
            tn += 1
        elif pred == 1 and y[i] == 0:
            fp += 1
        elif pred == 0 and y[i] == 1:
            fn += 1
    matrix = np.array([[tp, fp],
                        [fn, tn]])
    return matrix



## Results: 
# print("="*20+" Results "+"="*20)

# print(f"confusion matrix:\n{confusion_matrix(w_final, b_final, x_train, y_train)}")

# print("-"*40)

# accuracy = (confusion_matrix(w_final, b_final, x_train, y_train)[0][0] + confusion_matrix(w_final, b_final, x_train, y_train)[1][1])/len(x_train)
# print(f"Accuracy: {accuracy*100}%")

# print("-"*40)

# precision = confusion_matrix(w_final, b_final, x_train, y_train)[0][0]/(confusion_matrix(w_final, b_final, x_train, y_train)[0][0] + confusion_matrix(w_final, b_final, x_train, y_train)[0][1])
# print(f"Precision: {precision*100}%")

# print("-"*40)

# recall = confusion_matrix(w_final, b_final, x_train, y_train)[0][0]/(confusion_matrix(w_final, b_final, x_train, y_train)[0][0] + confusion_matrix(w_final, b_final, x_train, y_train)[1][0])
# print(f"Recall: {recall*100}%")

# print("-"*40)

# f1_score = 2 * (precision * recall) / (precision + recall)
# print(f"F1 Score: {f1_score*100}%")

# print("-"*40)