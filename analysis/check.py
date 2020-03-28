import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import getfiles as g
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize


params = [0.01, 0.15, 0, 0]


def sir(mydf, alpha=.5, beta=2, gamma=.001, mu=0.0001):

    print('\n---------Doing a SIR---------')
    dt = 0.01
    fill = int(1/dt)
    Nsteps = mydf.shape[0]
    N = 37.59E+6
    I0, R0 = 1, 1
    S0 = N - I0 - R0
    S = np.zeros(Nsteps)
    I = np.zeros(Nsteps)
    R = np.zeros(Nsteps)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    def ode(s, i, r):
        dS = -beta * (s * i) / N + gamma * r
        dI = beta * (s * i) / N - alpha * i - mu * i
        dR = alpha * i - gamma * r
        return dt * dS, dt * dI, dt * dR

    for n in np.arange(1, Nsteps):
        stemp = S[n - 1]
        itemp = I[n - 1]
        rtemp = R[n - 1]
        for k in range(0, fill):
            ds, di, dr = ode(stemp, itemp, rtemp)
            stemp = stemp + ds
            itemp = itemp + di
            rtemp = rtemp + dr
        S[n] = stemp
        I[n] = itemp
        R[n] = rtemp

    t = np.arange(Nsteps)
    return S, I, R


def getsir(params):

    df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk = g.main()
    mydf = df_ca
    I = mydf.loc[:, 'confirmed'].values
    R = mydf.loc[:, 'cured'].values
    print(R[6])
    N = np.full(len(I), 37.59E+6)
    S = N - I
    t = range(0, len(S))

    # -------------check
    mat = pd.DataFrame(list(zip(S, I, R)), columns=['S', 'I', 'R'])
    print(mat)

    # -------------plot things
    fig, axs = plt.subplots(2)
    fig.suptitle('Wuhan')
    plt.figure(11)
    #axs[0].set_yscale('log')
    #axs[1].set_yscale('log')
    #axs[0].plot(t, S, label='S')
    axs[0].plot(t, I, label='I')
    axs[0].plot(t, R, label='R')
    axs[0].legend()
    Shat, Ihat, Rhat = sir(mydf, *params)
    #axs[1].plot(t, Shat, label='Shat')
    axs[1].plot(t, Ihat, label='Ihat')
    axs[1].plot(t, Rhat, label='Rhat')
    plt.title('Doing a SIR')
    axs[1].legend()
    fig.show()


getsir(params)
