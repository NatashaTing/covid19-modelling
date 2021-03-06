import getfiles as g
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


# this is the casual exponential fit sheet

def exponential(x, k, x0):
    return np.exp(k * (x - x0))


def fit(df, series):

    xdata = np.arange(0, df.shape[0])
    ydata = df.loc[:, series].values
    p0 = [0.4, 1]
    fig = plt.figure()
    plt.plot(xdata, ydata, 'o', alpha=0.3, label='samples')
    plt.show()

    for i in range(7, len(xdata)):
        popt, pcov = curve_fit(exponential, xdata[i-6:i], ydata[i-6:i], p0)
        estimated_k, estimated_x0 = popt
        a = np.log(2)/estimated_k
        print('Day {} - {}: doubles in every {} days'.format(str(i-6), str(i), str(round(a, 2))))

    popt, pcov = curve_fit(exponential, xdata, ydata, p0, sigma=None)
    estimated_k, estimated_x0 = popt
    a = np.log(2)/estimated_k
    print('All time average: doubles in every {} days'.format(str(round(a, 2))))
    y_fitted = exponential(xdata, estimated_k, estimated_x0)

    plt.plot(xdata, y_fitted, '--', label='fitted')
    plt.ylim(0, 6E+4)
    plt.title('{} Confirmed Cases---Sigmoid Fit'.format(series))
    plt.legend()
    plt.show()


df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk = g.main()
fit(df_cn2, 'confirmed')