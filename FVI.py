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

NUM_FEATURES = 6
NUM_ACTIONS = 1
theta = np.zeros(shape=(NUM_FEATURES,1))
SAMPLE_SIZE = 400 #samples per iteration
ACTIONS_SAMPLES_SIZE = 50
EPOCHS = 100
DISCOUNT = 0.99
ACTIONS = [-1,0,1]

if __name__=="__main__":

    with open("samples.csv",'r') as f:
        reader = csv.reader(f)
        # next(reader)
        for line in reader:
            s_i.append(phi(json.loads(line[0])))
            a_i.append([int(line[1])])

    s_i1 = s_i[1:]
    s_i = s_i[:-1]
    a_i = a_i[:-1]
    scaler_s = StandardScaler()

    s_i = scaler_s.fit_transform(s_i)
    s_i1 = scaler_s.transform(s_i1)

    joblib.dump(scaler_s,"scaler_s")


    X = np.concatenate([np.array(s_i),np.array(a_i)],axis=1)
    Y = np.array(s_i1)
    C = np.linalg.inv(X.T @ X) @ X.T @ Y
    A = C[:NUM_FEATURES,:].T
    B = C[NUM_FEATURES:,:].T


    As = A@np.array(s_i).T
    Ba = B@np.array(a_i).T
    covariance = np.cov(As + Ba ,np.array(s_i1).T)[:NUM_FEATURES,NUM_FEATURES:]

    def reward(state):
        return 100 if -math.pi/9<state[2]<math.pi/9 else 0

    def next_state(state,action):
        state = np.array(state).reshape((NUM_FEATURES,1))

        a = A@state + B*action# + np.array(np.random.multivariate_normal((0,0,0,0,0,0),covariance)).reshape((NUM_FEATURES,1))
        return a

    for Epoch in range(EPOCHS):
        print(f"Epoch {Epoch+1}")
        samples = random.sample(list(range(len(s_i))), k=SAMPLE_SIZE)
        y = {}
        for i in samples:
            Q = []
            for a in ACTIONS:
                q = 1/ACTIONS_SAMPLES_SIZE * sum(reward(s_i[i]) + DISCOUNT* theta.T@next_state(s_i[i],a) for _ in range(ACTIONS_SAMPLES_SIZE))
                Q.append(q)
            y[i]=max(Q)
        Y = np.array([y[i] for i in samples]).T.reshape((1,-1))
        S = np.array([s_i[i] for i in samples]).T
        theta = np.linalg.inv(S@S.T)@S@Y.T

    np.save("A.npy",A)
    np.save("B.npy",B)
    np.save("Theta.npy",theta)