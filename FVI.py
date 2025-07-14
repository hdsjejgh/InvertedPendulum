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
theta = np.zeros(shape=(NUM_FEATURES,1))
SAMPLE_SIZE = 200 #samples per iteration
ACTIONS_SAMPLES_SIZE = 25
EPOCHS = 50
DISCOUNT = 0.975
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

    import collections

    print(collections.Counter(a[0] for a in a_i))
    scaler_s = StandardScaler()
    og_s_i = s_i


    s_i = scaler_s.fit_transform(s_i)
    s_i1 = scaler_s.transform(s_i1)

    joblib.dump(scaler_s,"scaler_s")


    X = np.concatenate([np.array(s_i),np.array(a_i)],axis=1)
    Y = np.array(s_i1)
    C = np.linalg.pinv(X.T @ X) @ X.T @ Y
    A = C[:NUM_FEATURES,:].T
    B = C[NUM_FEATURES:,:].T


    As = A@np.array(s_i).T
    Ba = B@np.array(a_i).T
    covariance = np.cov(As + Ba ,np.array(s_i1).T)[:NUM_FEATURES,NUM_FEATURES:]


    def reward(ind):
        x = og_s_i[ind][1]
        t = og_s_i[ind][3]
        dtheta = og_s_i[ind][6]

        upright_bonus = 100 if abs(t) < math.pi / 18 else 0
        position_penalty = -0.05 * abs(x)
        spin_penalty = -2.0 * abs(dtheta)

        return upright_bonus + position_penalty + spin_penalty

    def next_state(state,action):
        state = np.array(state).reshape((NUM_FEATURES,1))

        a = A @ state + B @ np.array([[action]])  + np.array(np.random.multivariate_normal([0]*NUM_FEATURES,covariance)).reshape((NUM_FEATURES,1))
        return a

    for Epoch in range(EPOCHS):
        print(f"Epoch {Epoch+1}")
        samples = random.sample(list(range(len(s_i))), k=SAMPLE_SIZE)
        y = {}
        for i in samples:
            Q = []
            for a in ACTIONS:
                q = 1/ACTIONS_SAMPLES_SIZE * sum(reward(i) + DISCOUNT* theta.T@next_state(s_i[i],a) for _ in range(ACTIONS_SAMPLES_SIZE))
                Q.append(q)

            y[i]=max(Q)
        Y = np.array([y[i] for i in samples]).T.reshape((1,-1))
        S = np.array([s_i[i] for i in samples]).T
        theta = np.linalg.pinv(S @ S.T + 1e-4 * np.eye(S.shape[0])) @ S @ Y.T

    np.save("A.npy",A)
    np.save("B.npy",B)
    np.save("Theta.npy",theta)