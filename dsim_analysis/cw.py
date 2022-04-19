import numpy as np
import config

class cw_analysis():

    def linearity():
        pass

    def scale():
        pass

    async def run(samples):
        hist = []
        cw_min = []
        cw_max = []

        for pol in range(len(samples)):            
            hist.append(np.histogram(samples[pol],config.hist_res))
            cw_max.append(np.max(samples[pol]))
            cw_min.append(np.min(samples[pol]))

        return cw_max, cw_min, hist