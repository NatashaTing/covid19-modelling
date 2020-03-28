import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import getfiles as g
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize


def sigmoidfit(mydf):

    print('\n---------Doing a sigmoid---------')
    def sigmoid(x, k, x0, L):
        return L / (1 + np.exp(-k * (x - x0)))

    xdata = np.arange(0, mydf.shape[0])
    ydata = mydf.loc[:, 'confirmed'].values
    p0 = [1, 0, max(ydata)]

    popt, pcov = curve_fit(sigmoid, xdata[:], ydata[:], p0,
                           bounds=([0, -np.inf, min(ydata)],
                                   [np.inf, np.inf, 3 * max(ydata)]))
    estimated_k, estimated_x0, est_L = popt
    print('popt is: ', popt, sep='\n')

    y_fitted = sigmoid(xdata, estimated_k, estimated_x0, est_L)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xdata, ydata, 'o', alpha=0.3, label='samples')
    ax.plot(xdata, y_fitted, '--', label='fitted')
    plt.title('Wuhan Confirmed Cases---Sigmoid Fit')
    ax.legend()
    return popt, pcov


def geterror(params):
    S, I, R, Shat, Ihat, Rhat = getsir(params)
    print('params', params)
    error = 0
    for t in range(0, len(S)):
        es = (S[t] - Shat[t])**2
        ei = (I[t] - Ihat[t])**2
        er = (R[t] - Rhat[t])**2
        error = error + (es + ei + er)
    error = error/(3*len(S))
    print(error/1000000)
    return error


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


def optimise():

    print('\n---------Doing an optimise---------')
    param0 = [0.01, 0.15, 0, 0]   # this was a good guess for Canada data
    popt, pcov = minimize(geterror, param0)
    estalpha, estbeta, estgamma, estmu = popt
    print('popt is: ', popt, sep='\n')
    S, I, R, Shat, Ihat, Rhat = getsir(popt)
    fig = plt.figure()
    xdata = range(0, len(S))
    ax = fig.add_subplot(111)
    ax.plot(xdata, I, 'o', alpha=0.3, label='samples')
    ax.plot(xdata, Ihat, '--', label='fitted')
    plt.title('Wuhan Confirmed Cases---Sigmoid Fit')
    ax.legend()
    ax.show()

    return popt, pcov


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



def linlog_fit(mydf, mytitle):

    print('\n---------Doing a linlog---------')
    y_pre = mydf.confirmed
    logy_pre = np.log10(y_pre)
    x_pre = np.arange(0, mydf.shape[0])
    myx = x_pre
    myy = logy_pre
    slope, intercept, r_value, p_value, std_err = linregress(myx, myy)
    print('number of periods:', mydf.shape[0])
    print("slope: {}, \ninterceipt: {}, \nr_value: {}".format(slope, intercept, r_value))
    plt.scatter(myx, myy)
    plt.plot(myx, myx * slope + intercept)
    plt.xlabel('date')
    plt.ylabel('log(confirmed)')
    plt.title('Fit on {}'.format(mytitle))
    return slope, intercept, r_value, p_value, std_err, plt


def main():
    df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk = g.main()
    slope, intercept, r_value, p_value, std_err, plt = linlog_fit(df_wuhan, 'only pre-quarantine')
    popt, pcov = sigmoidfit(df_wuhan)
    popt, pcov = optimise()

if __name__ == "__main__":
    main()