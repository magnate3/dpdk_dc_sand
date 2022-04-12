import allantools
# import matplotlib.pyplot as plt
import numpy as np
import config

def allan_variance(samples):

    n= config.n #3000
    taus = np.linspace(1,n,n)

    (t2, ad, ade, adn) = allantools.oadev(samples, rate=1, data_type="freq", taus=taus)
    # plt.plot(t2,ad)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.xlabel('WGN')
    # plt.ylabel('Allan Deviation')
    # plt.show()
    
    return t2, ad