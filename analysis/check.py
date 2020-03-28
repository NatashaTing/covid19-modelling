import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import getfiles as g
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize

params = [0.1, 0.4, 0.2]


def getsir(params):

    df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk = g.main()
    mydf = df_wuhan
    N = 11.08E+6
    I = mydf.loc[:, 'confirmed'].values
    R = mydf.loc[:, 'cured'].values
    S = N - I - R
    t = range(0, len(S))
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    plt.figure(11)
    axs[0].plot(t, S)
    axs[0].plot(t, I)
    axs[0].plot(t, R)
    plt.title('Doing a SIR')
    Shat, Ihat, Rhat = sir(mydf, *params)
    axs[1].plot(t, Shat)
    axs[1].plot(t, Ihat)
    axs[1].plot(t, Rhat)
    plt.title('Doing a SIR')
    plt.show()


# def sir(mydf, alpha=.5, beta=2, gamma=.001, mu=0.0001):
#
#     print('\n---------Doing a SIR---------')
#     dt = 0.01
#     fill = int(1/dt)
#     Nsteps = mydf.shape[0]
#     N = 100
#     I0, R0 = 1, 1
#     S0 = N - I0 - R0
#     S = np.zeros(Nsteps)
#     I = np.zeros(Nsteps)
#     R = np.zeros(Nsteps)
#     S[0] = S0
#     I[0] = I0
#     R[0] = R0
#
#     def ode(s, i, r):
#         dS = -beta * (s * i) / N + gamma * r
#         dI = beta * (s * i) / N - alpha * i - mu * i
#         dR = alpha * i - gamma * r
#         return dt * dS, dt * dI, dt * dR
#
#     for n in np.arange(1, Nsteps):
#         stemp = S[n - 1]
#         itemp = I[n - 1]
#         rtemp = R[n - 1]
#         for k in range(0, fill):
#             ds, di, dr = ode(stemp, itemp, rtemp)
#             stemp = stemp + ds
#             itemp = itemp + di
#             rtemp = rtemp + dr
#         S[n] = stemp
#         I[n] = itemp
#         R[n] = rtemp
#
#     t = np.arange(Nsteps)
#     plt.figure()
#     plt.plot(t, S, label = 'Sus')
#     plt.plot(t, I, label = 'Infected')
#     plt.plot(t, R, label = 'Recov')
#     plt.legend()
#     plt.show()
#     print(S, I, R, sep='\n')
#     return S, I, R
