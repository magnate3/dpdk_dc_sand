import numpy as np
import config
import allan_var

class wgn_analysis():

    async def run(samples):

        mean = []
        std_dev = []
        var = []
        hist = []
        allan_dev = []

        for pol in range(len(samples)):            
            mean.append(np.mean(samples[pol]))
            std_dev.append(np.std(samples[pol]))
            var.append(np.var(samples[pol]))
            hist.append(np.histogram(samples[pol],config.hist_res))
            allan_dev.append(allan_var.allan_variance(samples[pol]))

        return mean, std_dev, var, hist, allan_dev