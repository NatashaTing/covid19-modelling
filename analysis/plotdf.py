
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
    print(df.shape)
    try:
        title = "%s, %s, %s" % (df.loc[0, 'countryCode'], df.loc[0, 'provinceCode'], df.loc[0, 'cityCode'])
        fig.suptitle(title)
    except KeyError:
        fig.suptitle(df.loc[0, 'countryCode'])

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
    print('this done.------')
    pngtitle = '{}.png'.format(df.loc[0, 'countryCode'])
    fig.savefig(pngtitle)
print('hello')
