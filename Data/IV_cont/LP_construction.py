import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import truncnorm

# ============================================================
# Continuous IV Data Generator
# ============================================================

def generate_data_IV(n, lam):
    def mix_noise():
        return lam * truncnorm.rvs(-1,1,size=(n,1)) + \
               (1-lam) * np.random.uniform(-1,1,(n,1))

    EY = mix_noise()
    U  = mix_noise()
    Z  = mix_noise()

    T_prob = (1.5 + Z + 0.5*U)/3
    T = np.random.binomial(1, np.clip(T_prob,0,1))

    Y = 3*T - 1.5*T*U + U + EY
    return np.hstack([Z,T,Y])

# ============================================================
# Discretization
# ============================================================

def discretize(x,k):
    bins = np.linspace(x.min(),x.max(),k+1)
    return np.clip(np.digitize(x,bins)-1,0,k-1)

# ============================================================
# Empirical P(T,Y | Z)
# ============================================================

def empirical_distribution_IV(data,k):
    Zb = discretize(data[:,0],k)
    T  = data[:,1].astype(int)
    Yb = discretize(data[:,2],k)

    P = np.zeros((k,2,k))
    count = np.zeros(k)

    for z,t,y in zip(Zb,T,Yb):
        P[z,t,y]+=1
        count[z]+=1

    for z in range(k):
        if count[z]>0:
            P[z,:,:]/=count[z]

    return P

# ============================================================
# LP system
# ============================================================

def latent_types(k):
    T_types = list(product([0,1], repeat=k))
    Y_types = list(product(range(k), repeat=2))
    return T_types, Y_types


def build_constraints_IV(P,k):

    T_types,Y_types = latent_types(k)
    n_latent = len(T_types)*len(Y_types)

    A=[]
    b=[]
    labels=[]

    for z in range(k):
        for t in [0,1]:
            for y in range(k):

                row=np.zeros(n_latent)
                idx=0

                for Tt in T_types:
                    for Yt in Y_types:
                        if Tt[z]==t and Yt[t]==y:
                            row[idx]=1
                        idx+=1

                A.append(row)
                b.append(P[z,t,y])
                labels.append((z,t,y))

    # normalization constraint
    A.append(np.ones(n_latent))
    b.append(1.0)
    labels.append(("norm",0,0))

    c=[Yt[1]-Yt[0] for Tt in T_types for Yt in Y_types]

    return np.array(A), np.array(b), np.array(c), labels