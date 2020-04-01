# this is a sheet to simulate the SIR model and inspect its fit with the data
# lots of plots in this sheet to visually inspect fit
import matplotlib.pyplot as plt
import getfiles as g



df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk = g.main()

print(df_wuhan.head(1))
for df in g.main():
    df.reset_index(inplace=True)
    x = range(0, df.shape[0])
    sus = df.loc[:, 'suspected']
    conf = df.loc[:, 'confirmed']
    recov = df.loc[:, 'cured']
    dead = df.loc[:, 'dead']
    fig, axs = plt.subplots(2, 2)
    print('-----')
    try:
        if not str(df.loc[0, 'provinceCode']) == 'nan':
            title = "%s %d %d" % (df.loc[0, 'countryCode'], int(df.loc[0, 'provinceCode']), int(df.loc[0, 'cityCode']))
            fig.suptitle(title)
        else:
            title = "%s" % (df.loc[0, 'countryCode'])
            fig.suptitle(df.loc[0, 'countryCode'])
    except KeyError:
        title = "%s" % (df.loc[0, 'countryCode'])
        fig.suptitle(df.loc[0, 'countryCode'])
    print(title)
    axs[0, 0].plot(x, sus)
    axs[0, 0].set_title('suspected')
    axs[0, 1].plot(x, conf, 'tab:orange')
    axs[0, 1].set_title("confirmed")
    axs[1, 0].plot(x, recov, 'tab:green')
    axs[1, 0].set_title("cured")
    axs[1, 1].plot(x, dead, 'tab:red')
    axs[1, 1].set_title("dead")
    #for ax in axs.flat:
    #    ax.set(xlabel='x-label', ylabel='y-label')
    #for ax in axs.flat:
    #    ax.label_outer()
    fig.show()
    print('------')
    path = 'plots/' + title
    fig.savefig(title)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import getfiles as g
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize

params = np.array(9.46, 7.76145579E-2)
#params = [0.71534957, 0.15000557, 0., 0.7088634]
#
# def geterror(params):
#     S, I, R, D, Shat, Ihat, Rhat, Dhat = getsir(params)
#     print('params', params)
#     error = 0
#     for t in range(0, len(S)):
#         ei = (I[t] - Ihat[t])**2
#         er = (R[t] - Rhat[t])**2
#         ed = (D[t] - Dhat[t])**2
#         error = error + (ei + er + ed)
#     error = error/(3*len(S))
#     print(error/1000000)
#     return error
#

def getsir(params):

    df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk = g.main()
    mydf = df_wuhan
    I = mydf.loc[:, 'confirmed'].values
    R = mydf.loc[:, 'cured'].values
    D = mydf.loc[:, 'dead'].values

    N = 37.59E+6
    I = (I - R - D) / N
    S = 1 - I
    R = R/N
    D = D/N
    t = range(0, len(S))


    Shat, Ihat, Rhat = sir(mydf, *params)
    Dhat = 1 - Shat - Ihat - Rhat
    print('maxI:{}, lastDeath:{}'.format(max(I), D[-1]))
    print('maxIhat:{}, lastDHat:{}'.format(max(Ihat), Dhat[-1]))

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
    fig.savefig('plots/Wuhan_fitted.png')
    #
    # plt.figure()
    # plt.plot(t, I, label='I', color='r')
    # plt.plot(t, R, label='R', color='g')
    # plt.plot(t, D, label='D', color='k')
    # plt.plot(t, Ihat, label='Ihat', linestyle='--', color='r')
    # plt.plot(t, Rhat, label='Rhat', linestyle='--', color='g')
    # plt.plot(t, Dhat, label='Dhat', linestyle='--', color='k')
    # plt.ylim(1E-10,0.002)
    # #plt.yscale('log')
    # plt.legend()
    # plt.show()

    #plt.figure()
    #plt.plot(t, I, label='I')
    #plt.plot(t, Ihat, label='Ihat')
    #plt.legend()
    #plt.show()
    # -------------check
    #mat = pd.DataFrame(list(zip(S, Shat, I, Ihat, R, Rhat, D, Dhat)), columns=['S', 'Shat', 'I', 'Ihat', 'R', 'Rhat', 'D', 'Dhat'])
    #print(mat)

    return S, I, R, D, Shat, Ihat, Rhat, Dhat

#
# def optimise():
#
#     print('\n---------Doing an optimise---------')
#     param0 = [0.01, 0.15, 0, 0]   # this was a good guess for Canada data
#     popt, pcov = minimize(geterror, param0,
#                           bounds=((0, None), (0, None), (0, None), (0, None)))
#     estalpha, estbeta, estgamma, estmu = popt
#     print('popt is: ', popt, sep='\n')
#     S, I, R, Shat, Ihat, Rhat = getsir(popt)
#     fig = plt.figure()
#     xdata = range(0, len(S))
#     ax = fig.add_subplot(111)
#     ax.plot(xdata, I, 'o', alpha=0.3, label='samples')
#     ax.plot(xdata, Ihat, '--', label='fitted')
#     plt.title('Wuhan Confirmed Cases---Sigmoid Fit')
#     ax.legend()
#     ax.show()
#
#     return popt, pcov


def sir(mydf, alpha, beta, gamma, mu):

    print('\n---------Doing an SIR---------')
    dt = 0.1
    fill = int(1/dt)
    Nsteps = mydf.shape[0]
    N = 37.59E+6
    I0, R0 = 1/N, 0
    S0 = 1 - I0 - R0
    S = np.zeros(Nsteps)
    I = np.zeros(Nsteps)
    R = np.zeros(Nsteps)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    def ode(s, i, r):
        dS = -beta * (s * i) + gamma * r
        dI = beta * (s * i) - alpha * i - mu * i
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


df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk = g.main()
getsir(params)
