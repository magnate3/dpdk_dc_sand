import numpy as np
import config

class cw_analysis():

    async def run(samples):
        hist = []

        for pol in range(len(samples)):            
            hist.append(np.histogram(samples[pol],config.hist_res))

        return hist