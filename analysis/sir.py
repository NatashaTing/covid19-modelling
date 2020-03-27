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


def optimise(mydf):

    print('\n---------Doing an optimise---------')
    params = [0.4, 0.1, 0.2]
    Shat, Ihat, Rhat = sir(mydf, *params)
    S = mydf.loc[:, 'suspected']
    I = mydf.loc[:, 'confirmed']
    R = mydf.loc[:, 'cured']
    xdata = range(0, mydf.shape[0])

    def errors(S, I, R, Shat, Ihat, Rhat):
        error = 0
        for time in range(0, mydf.shape[0]):
            es = (S - Shat)**2
            ei = (I - Ihat)**2
            er = (R - Rhat)**2
            error = error + (es + ei + er)
        return error

    etots = errors(S, I, R, Shat, Ihat, Rhat)
    popt, pcov = minimize(errors, S, I, R, *params)
    estShat, estIhat, estRhat = popt
    print('popt is: ', popt, sep='\n')

    y_fitted = errors(S, I, R, estShat, estIhat, estRhat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xdata, S, 'o', alpha=0.3, label='samples')
    ax.plot(xdata, y_fitted, '--', label='fitted')
    plt.title('Wuhan Confirmed Cases---Sigmoid Fit')
    ax.legend()
    ax.show()

    return popt, pcov


def sir(mydf, alpha=.4, beta=.1, gamma=.2, mu=0):

    print('\n---------Doing a SIR---------')
    dt = 0.0001
    Nsteps = mydf.shape[0]
    N = 100
    I0, R0 = 1, 1
    S0 = N - I0 - R0
    S = np.zeros(Nsteps)
    I = np.zeros(Nsteps)
    R = np.zeros(Nsteps)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    def ode(s, i, r):
        dS = -beta * s * i + gamma * r
        dI = beta * s * i - alpha * i
        # dI = beta * s * i - alpha * i - mu * i
        dR = alpha * i - gamma * r
        return dt * dS, dt * dI, dt * dR

    for n in np.arange(1, Nsteps):
        V = [S[n - 1], I[n - 1], R[n - 1]]
        S[n], I[n], R[n] = map(sum, zip(V, ode(S[n - 1], I[n - 1], R[n - 1])))

    t = dt * np.arange(Nsteps)

    plt.figure()
    plt.plot(t, S, label='Susp')
    plt.plot(t, I, label='Inft')
    plt.plot(t, R, label='Recv')
    plt.title('SIR')
    plt.legend()
    plt.show()

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
    popt, pcov = optimise(df_wuhan)

if __name__ == "__main__":
    main()