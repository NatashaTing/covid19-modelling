import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import getfiles as g
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize

df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk = g.main()
mydf = df_wuhan

# this is a scratch sheet for estimating parameters with identification used by
# Anastassopoulo et al at https://doi.org/10.1101/2020.02.11.20022186.

def geterror(params):
    print('Begin geterror')
    S, I, R, D, Shat, Ihat, Rhat, Dhat = getsir(params)
    print('params', params)
    error = 0
    error += (max(I)-max(Ihat))**2
    error += (max(R)-max(Rhat))**2
    error += (max(D)-max(Dhat))**2
    #for t in range(0, len(S)):
    #    ei = (I[t] - Ihat[t])**2
    #    er = (R[t] - Rhat[t])**2
    #    ed = (D[t] - Dhat[t])**2
    #    error = error + (ei + er + ed)

    print('error=', error)
    return error


def getsir(params):
    I = mydf.loc[:, 'confirmed'].values
    R = mydf.loc[:, 'cured'].values
    D = mydf.loc[:, 'dead'].values
    N = 11.8E+8
    print('getsir N', N)
    #t = range(0, len(I))
    I = (I - R - D) / N

#    S = 1 - I

    R = R/N
    D = D/N
    S = 1 - I - R - D

    Shat, Ihat, Rhat = sir(*params)
    Dhat = 1 - Shat - Ihat - Rhat


    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(t, S, label='S')
    # axs[0, 0].plot(t, Shat, linestyle='--', label='Shat')
    # axs[0, 0].set_title('Susceptible')
    # axs[0, 0].legend()
    #
    # axs[0, 1].plot(t, I, label='I')
    # axs[0, 1].plot(t, Ihat, linestyle='--', label='Ihat')
    # axs[0, 1].set_title('Infected')
    # axs[0, 1].legend()
    #
    # axs[1, 0].plot(t, R, label='R')
    # axs[1, 0].plot(t, Rhat, linestyle='--', label='Rhat')
    # axs[1, 0].set_title('Recovered')
    # axs[1, 0].legend()
    #
    # axs[1, 1].plot(t, D, label='D')
    # axs[1, 1].plot(t, Dhat, linestyle='--', label='Dhat')
    # axs[1, 1].set_title('Dead')
    # axs[1, 1].legend()
    #
    # fig.suptitle('Wuhan')
    # fig.show()
    # fig.savefig('plots/Wuhan_fitted.png')

    return S, I, R, D, Shat, Ihat, Rhat, Dhat




def sir(beta, mu, I0):
    print('--------------Doing an SIR-----------')
    dt = 0.1
    alpha=1
    fill = int(1/dt)
    Nsteps = mydf.shape[0]
    #N = 37.59E+6
    R0 = 0                     # put in sensible, then check that output fits the first non-zero value
    S0 = 1 - I0 - R0
    S = np.zeros(Nsteps)
    I = np.zeros(Nsteps)
    R = np.zeros(Nsteps)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    def ode(s, i, r):
        dS = -beta * (s * i)
        dI = beta * (s * i) - alpha * i - mu * i
        dR = alpha * i
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


def get_beta_mu(mydf):

    I = mydf.loc[:, 'confirmed'].values
    R = mydf.loc[:, 'cured'].values
    D = mydf.loc[:, 'dead'].values
    # D(t) = hatgamma(I(t))

    res = linregress(I, D)
    est_beta = res[0]

    res = linregress(I, R)
    est_gamma = res[0]

    print(est_beta, est_gamma)
    return est_beta, est_gamma


def optimise(mydf):

    N = 11E+6
    Nbound = N/100
    est_beta, est_gamma = get_beta_mu(mydf)
    param0 = [0.12]
    bound_1 = (0,10)
    res = minimize(geterror, param0, method='L-BFGS-B', bounds=bound_1)
    print('popt is: ', res, sep='\n')
    params = [popt.x, est_beta, est_gamma]

    S, I, R, D, Shat, Ihat, Rhat, Dhat = getsir(params)
    fig = plt.figure()
    xdata = range(0, len(S))
    Shat, Ihat, Rhat = sir(*params)
    Dhat = 1 - Shat - Ihat - Rhat
    print('maxI:{}, lastDeath:{}'.format(max(I), D[-1]))
    print('maxIhat:{}, lastDHat:{}'.format(max(Ihat), Dhat[-1]))

    t = range(0, len(S))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(t, S, label='S')
    axs[0, 0].plot(t, Shat, linestyle='--', label='Shat')
    axs[0, 0].set_title('Susceptible')
    axs[0, 0].legend()

    axs[0, 1].plot(t, I, label='I')
    axs[0, 1].plot(t, Ihat, linestyle='--', label='Ihat')
    axs[0, 1].set_title('Infected')
    axs[0, 1].legend()

    axs[1, 0].plot(t, R, label='R')
    axs[1, 0].plot(t, Rhat, linestyle='--', label='Rhat')
    axs[1, 0].set_title('Recovered')
    axs[1, 0].legend()

    axs[1, 1].plot(t, D, label='D')
    axs[1, 1].plot(t, Dhat, linestyle='--', label='Dhat')
    axs[1, 1].set_title('Dead')
    axs[1, 1].legend()

    fig.suptitle('Wuhan')
    fig.show()


optimise(df_wuhan)
