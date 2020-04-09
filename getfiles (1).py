# this is the sheet to import dataframes of global COVID-19 data
import pandas as pd
import sys
import os


def getsums(df, regionname):

    dates = df['date'].unique()
    howmany = dates.shape[0]
    regionlist = [regionname for i in range(howmany)]
    confirmed = df.groupby("date").confirmed.sum()
    suspected = df.groupby("date").suspected.sum()
    cured = df.groupby("date").cured.sum()
    dead = df.groupby("date").dead.sum()
    cols = ['date', 'countryCode', 'confirmed', 'suspected', 'cured', 'dead']

    if len(dates) == len(regionlist) == len(confirmed) == len(suspected) == len(cured) == len(dead):
        try:
            newindex = range(0, howmany)
            newdf = pd.DataFrame(list(zip(dates, regionlist, confirmed, suspected, cured, dead)),
                                 columns=cols, index=newindex)
            return newdf
        except:
            print('Error creating newdf')
    else:
        print('no df to return')


def main():

    cols = ['date', 'countryCode', 'provinceCode', 'cityCode', 'confirmed',
            'suspected', 'cured', 'dead']
    print('cwd =', os.getcwd())

    if os.getcwd() == '/Users/zeptinc/Google Drive/UAlberta/Wi2020/MATH371/Project/data-changhailan':
        df = pd.read_csv('Wuhan-2019-nCov.csv')
    elif os.getcwd() == '/Users/zeptinc/Google Drive/UAlberta/Wi2020/MATH371/Project':
        df = pd.read_csv('data-changhailan/Wuhan-2019-nCov.csv')

    df_cn = df[df['countryCode'] == 'CN'].loc[:, cols]
    df_hubei = df[df['provinceCode'] == 420000.0].loc[:, cols]
    df_wuhan = df[df['cityCode'] == '420100'].loc[:, cols]
    df_ca = df[df['countryCode'] == 'CA'].loc[:, cols]
    df_it = df[df['countryCode'] == 'IT'].loc[:, cols]
    df_sk = df[df['countryCode'] == 'KR'].loc[:, cols]
    df_sg = df[df['countryCode'] == 'SG'].loc[:, cols]
    df_uk = df[df['countryCode'] == 'GB'].loc[:, cols]

    df_cn2 = getsums(df_cn, 'CN')
    df_hubei2 = getsums(df_hubei, 'Hubei')

    return df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk


if __name__ == "__main__":
    df_cn2, df_hubei2, df_wuhan, df_ca, df_it, df_sk, df_sg, df_uk = main()