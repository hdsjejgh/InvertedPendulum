import csv
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
import random
import joblib
from sim import phi

s_i = []
a_i = []

NUM_FEATURES = 12
NUM_ACTIONS = 1
theta = np.zeros(shape=(NUM_FEATURES,1)) #theta is used to predict value given feature map
SAMPLE_SIZE = 400 #samples per iteration
ACTIONS_SAMPLES_SIZE = 25 #number of actions to sample to get predicted next
EPOCHS = 100
DISCOUNT = 0.975
ACTIONS = [-1,0,1] #left, stay, right

if __name__=="__main__":

    with open("samples.csv",'r') as f:
        reader = csv.reader(f)
        # next(reader)
        for line in reader:
            s_i.append(phi(json.loads(line[0])))
            a_i.append([int(line[1])])

    s_i1 = s_i[1:] #s_{i+1}. s_i[j] =s_i1[j-1]
    s_i = s_i[:-1] #states
    a_i = a_i[:-1] #actions for each s_i

    import collections

    print(collections.Counter(a[0] for a in a_i))
    scaler_s = StandardScaler()
    og_s_i = s_i #original states before being scaled

    #scales both the s_i and s_i1 arrays
    s_i = scaler_s.fit_transform(s_i)
    s_i1 = scaler_s.transform(s_i1)

    #saves the scaler
    joblib.dump(scaler_s,"scaler_s")

    #Calculates A & B (s_{i+1} = A@s_i + B@action) using least squares. Concatenate the vectors, do least squares, then get the appropriate parts of the resulting array C
    X = np.concatenate([np.array(s_i),np.array(a_i)],axis=1)
    Y = np.array(s_i1)
    C = np.linalg.pinv(X.T @ X) @ X.T @ Y
    A = C[:NUM_FEATURES,:].T
    B = C[NUM_FEATURES:,:].T


    As = A@np.array(s_i).T
    Ba = B@np.array(a_i).T
    covariance = np.cov(As + Ba ,np.array(s_i1).T)[:NUM_FEATURES,NUM_FEATURES:] #Calculates covariance between predictions and actual next states. used to get more accurate noise

    def reward(ind):
        return 100 if -math.pi/18<og_s_i[ind][3]<math.pi/18 else 0 #reward is 100 if pendulum is within 20 degrees of upright

    def next_state(state,action):
        state = np.array(state).reshape((NUM_FEATURES,1))
        #A@s_i + B@action + noise = s_{i+1}
        a = A @ state + B @ np.array([[action]])  + np.array(np.random.multivariate_normal([0]*NUM_FEATURES,covariance)).reshape((NUM_FEATURES,1))
        return a

    for Epoch in range(EPOCHS):
        print(f"Epoch {Epoch+1}")
        samples = random.sample(list(range(len(s_i))), k=SAMPLE_SIZE)
        #y stores the value of the optimal action of each state
        y = {}
        for i in samples:
            Q = [] #Stores value of each action
            for a in ACTIONS:
                #q(a=action) = 1/k  sum over k of [ R(s_i) + discount * theta^T @ (A@s_i + B@a) ]
                #(A@s_i + B@a) is prediction of next state
                #theta^T @ (A@s_i + B@a) is prediction of value of next state
                q = 1/ACTIONS_SAMPLES_SIZE * sum(reward(i) + DISCOUNT* theta.T@next_state(s_i[i],a) for _ in range(ACTIONS_SAMPLES_SIZE))
                Q.append(q)

            y[i]=max(Q)
        #Y gets all the best action's value for each sample and puts it in a row vector
        Y = np.array([y[i] for i in samples]).T.reshape((1,-1))

        S = np.array([s_i[i] for i in samples]).T
        #uses least squares to optimize theta in (theta^T @ phi(samples) - Y)**2
        theta = np.linalg.pinv(S @ S.T + 1e-4 * np.eye(S.shape[0])) @ S @ Y.T

    np.save("A.npy",A)
    np.save("B.npy",B)
    np.save("Theta.npy",theta)