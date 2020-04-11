import numpy as np
import matplotlib.pyplot as plt
import getfiles as g
from scipy.optimize import minimize, NonlinearConstraint
from scipy import interpolate
import pandas as pd

global I, R, D, title, population
df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk = g.main()
mydf = df_wuhan
title = 'Wuhan'
population = 1.18E+7
I = mydf.loc[:, 'confirmed'].values
R = mydf.loc[:, 'cured'].values
D = mydf.loc[:, 'dead'].values
I = I - R - D


def geterror(params):
    """"Returns total errors"""""
    print('params', params)
    S, I, R, D, Shat, Ihat, Rhat, Dhat = getsir(params)      # get key series
    N = params[-1]
    Imax = max(I)
    where = np.where(I == Imax)[0]
    where_hat = np.where(Ihat == max(Ihat))[0]
    error_i = ((Imax - max(Ihat))/Imax)**2
    k = ((where+1)/(where_hat+1))              # +1 to keep denominator non-zero
    t = np.arange(0, len(Rhat))                # these generated by simulated dt
    treal = np.arange(0, len(R))               # these are in days

    # need to interpolate to match simulated timescales
    def interp(Rhat, Dhat):
        f1 = interpolate.interp1d(t, Rhat)
        f2 = interpolate.interp1d(t, Dhat)
        Rhat = f1(treal / k)
        Dhat = f2(treal / k)
        return Rhat, Dhat

    Rhat, Dhat = interp(Rhat, Dhat)
    length = len(R)
    error_r = np.sum(np.square((R - Rhat)/max(R)))/length
    error_d = np.sum(np.square((D - Dhat)/max(D)))/length
    error = error_i + error_r + error_d        # total error
    print('error =', error)
    return error


def getsir(params):
    """"get estimated series from the SIR Model"""""
    global I, R, D
    N = params[-1]
    S = N - I - R - D
    Shat, Ihat, Rhat = sir(mydf, *params)
    Dhat = N - Shat - Ihat - Rhat
    return S, I, R, D, Shat, Ihat, Rhat, Dhat


def optimise():
    """"Optimises parameters to fit real S, I, R, D, data """""
    print('\n---------Doing an optimise---------')
    global I, R, D, title, population

    def constraintf(params):
        alpha = params[0]
        mu = params[1]
        N = params[2]
        return (alpha + mu - 1)*N

    Imax = max(I)
    where = np.where(I == Imax)[0]
    R_Imax = R[min(where)]
    D_Imax = D[min(where)]
    lb = -(Imax + R_Imax + D_Imax)
    ub = 0
    minpop = max(I + R + D)
    print(minpop)
    maxpop = population
    cons = NonlinearConstraint(constraintf, lb, ub)
    param0 = np.array([0.5, 0.1, 1E+5])                # Dependent on initial population condition
    # trust-constr, SLSQP
    popt = minimize(geterror, param0, bounds=((0, 1), (0, 1), (minpop, maxpop)), method='trust-constr', constraints=cons)
    print('popt is: ', popt, sep='\n')
    params=popt.x
    # params=[0.5,0.1,5E+4]
    S, I, R, D, Shat, Ihat, Rhat, Dhat = getsir(params)
    N0 = params[-1]
    Dhat = N0 - Shat - Ihat - Rhat
    Imax = max(I)
    where = np.where(I == Imax)[0]
    where_hat = np.where(Ihat == max(Ihat))[0]
    k = ((where + 1) / (where_hat + 1))

    # plot results to see fit

    t = np.arange(len(S))
    that = np.arange(len(Shat))*k

    print('maxI:{}, lastDeath:{}'.format(max(I), D[-1]))
    print('maxIhat:{}, lastDHat:{}'.format(max(Ihat), Dhat[-1]))

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(t, S, label='S')
    axs[0, 0].plot(that, Shat, linestyle='--', label='Shat')
    axs[0, 0].set_title('Susceptible')
    axs[0, 0].legend()
    axs[0, 0].set_xlim([0, len(S)])

    axs[0, 1].plot(t, I, label='I')
    axs[0, 1].plot(that, Ihat, linestyle='--', label='Ihat')
    axs[0, 1].set_title('Infected')
    axs[0, 1].legend()
    axs[0, 1].set_xlim([0, len(I)])

    axs[1, 0].plot(t, R, label='R')
    axs[1, 0].plot(that, Rhat, linestyle='--', label='Rhat')
    axs[1, 0].set_title('Recovered')
    axs[1, 0].legend()
    axs[1, 0].set_xlim([0, len(R)])

    axs[1, 1].plot(t, D, label='D')
    axs[1, 1].plot(that, Dhat, linestyle='--', label='Dhat')
    axs[1, 1].set_title('Dead')
    axs[1, 1].legend()
    axs[1, 1].set_xlim([0, len(D)])

    alpha = round(params[0]/k[0], 4)
    beta = round(1/k[0], 4)
    mu = round(params[1]/k[0], 4)
    N0 = round(params[-1], 4)
    r0 = beta/alpha

    xlab = ('α:{}, β:{}, $\mu$:{}, N0:{}, R0: {}, init: {}'.format(alpha, beta, mu, N0, r0, param0))
    print('alpha', alpha)
    print('beta', beta)
    print('mu', mu)
    print('N0', N0)
    print('r0', beta/alpha)
    # plt.xlabel(xlab)
    # fig.subplots_adjust(bottom=0.5)
    fig.text(0.5, 0.02, xlab, va='center', ha='center', fontsize=6, clip_on=True)
    plt.tight_layout()

    fig.suptitle(title)
    fig.show()

    return popt, k


def sir(mydf, alpha, mu, N):
    """"Calculate s, i, r, d from system of ODEs"""""

    print('\n---------Doing an SIR---------')
    dt = 0.1
    Nsteps = mydf.shape[0]*100
    # N = 37.59E+6  # 11 million Wuhan
    beta = 1
    I0, R0 = 1, 0
    S0 = N - I0 - R0
    S = np.zeros(Nsteps)
    I = np.zeros(Nsteps)
    R = np.zeros(Nsteps)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    # print('alpha, mu', alpha, mu)

    def ode(s, i, r):
        dS = -beta * (s * i) / N
        dI = beta * (s * i) / N - alpha * i - mu * i
        dR = alpha * i
        return dS, dI, dR

    for n in np.arange(1, Nsteps):
        s = S[n - 1]
        i = I[n - 1]
        r = R[n - 1]
        ds, di, dr = ode(s, i, r)
        S[n] = s + dt * ds
        I[n] = i + dt * di
        R[n] = r + dt * dr
    # t = np.arange(Nsteps)
    return S, I, R


def main():
    popt, k = optimise()


if __name__ == "__main__":
    main()