import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import getfiles as g
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize

#N = 37.59E+6
#params = np.array([1, 2E-2, 0, 0.001, N / 3]) * 3
#params = [0.71534957, 0.15000557, 0., 0.7088634]


def geterror(params):
    print('--------------Doing a geterror-----------')
    S, I, R, D, Shat, Ihat, Rhat, Dhat, Nhat = getsir(params)
    print('params', params)
    error = 0
    for t in range(0, len(S)):
        ei = (I[t] - Ihat[t])**2
        er = (R[t] - Rhat[t])**2
        ed = (D[t] - Dhat[t])**2
        error = error + (ei + er + ed)
    error = error/(3*len(S))
    print(error/1000000)
    return error


def getsir(params):
    print('--------------Doing a getsir-----------')
    df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk = g.main()
    mydf = df_wuhan
    I = mydf.loc[:, 'confirmed'].values
    R = mydf.loc[:, 'cured'].values
    D = mydf.loc[:, 'dead'].values
    N = params[-1]
    t = range(0, len(I))
    I = (I - R - D) / N

#    S = 1 - I

    R = R/N
    D = D/N
    S = 1 - I - R - D

    print('params:', params)
    Shat, Ihat, Rhat, Nhat = sir(mydf, *params)
    Dhat = 1 - Shat - Ihat - Rhat
    print('maxI:{}, lastDeath:{}'.format(max(I), D[-1]))
    print('maxIhat:{}, lastDHat:{}'.format(max(Ihat), Dhat[-1]))

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

    return S, I, R, D, Shat, Ihat, Rhat, Dhat, Nhat




def sir(mydf, alpha, beta, gamma, mu, N):
    print('--------------Doing an SIR-----------')
    dt = 0.1
    fill = int(1/dt)
    Nsteps = mydf.shape[0]
    #N = 37.59E+6
    I0, R0 = 1/N, 0
    S0 = 1 - I0 - R0
    S = np.zeros(Nsteps)
    I = np.zeros(Nsteps)
    R = np.zeros(Nsteps)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    def ode(s, i, r,i_r_delay,i_d_delay):
        dS = -beta * (s * i) + gamma * r
        dI = beta * (s * i) - alpha * i_r_delay - mu * i_d_delay
        dR = alpha * i_r_delay - gamma * r
        return dt * dS, dt * dI, dt * dR

    for n in np.arange(1, Nsteps):
        stemp = S[n - 1]
        itemp = I[n - 1]
        rtemp = R[n - 1]
        if n<10:
            i_r_delay=0
        else:
            if I[n]==0:
                i_r_delay=0
            else:
                i_r_delay=I[n-10]
        if n<5:
            i_d_delay=0
        else:
            if I[n]==0:
                i_d_delay=0
            else:
                i_d_delay=I[n-5]
        for k in range(0, fill):
            ds, di, dr = ode(stemp, itemp, rtemp,i_r_delay,i_d_delay)
            stemp = stemp + ds
            itemp = itemp + di
            rtemp = rtemp + dr
        S[n] = stemp
        I[n] = itemp
        R[n] = rtemp

    t = np.arange(Nsteps)
    return S, I, R, N


def optimise():

    print('\n---------Doing an optimise---------')
    N = 37.59E+6
    # params = [alpha, beta, gamma, mu, population/adjustment]*adjustment
    param0 = np.array([1, 2E-2, 0, 0.001, N / 6]) * 3
    # param0 = [0.01, 0.15, 0, 0]
    popt, pcov = minimize(geterror, param0,
                          bounds=((0, None), (0, None), (0, None),
                                  (0, None), (0, None)))
    estalpha, estbeta, estgamma, estmu, estn = popt
    print('popt is: ', popt, sep='\n')
    S, I, R, D, Shat, Ihat, Rhat, Dhat, Nhat = getsir(popt)
    fig = plt.figure()
    xdata = range(0, len(S))
    ax = fig.add_subplot(111)
    ax.plot(xdata, I, 'o', alpha=0.3, label='samples')
    ax.plot(xdata, Ihat, '--', label='fitted')
    plt.title('Wuhan Confirmed Cases---Sigmoid Fit')
    ax.legend()
    ax.show()

    return popt, pcov


df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk = g.main()
optimise()

#df_wuhan.to_csv('df_wuhan.csv')