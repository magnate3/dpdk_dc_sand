import numpy as np
import config
import allan_var

# wgn_mean_p0 = []
# wgn_std_dev_p0 = []
# wgn_var_p0 = []

# wgn_mean_p1 = []
# wgn_std_dev_p1 = []
# wgn_var_p1 = []

class wgn_analysis():
    # def __init__(self) -> None:
    #     self.mean = []
    #     self.std_dev = []
    #     self.var = []
    #     self.hist = []
    #     self.allan_var = []

    #     for _ in range(config.N_POLS):
    #         self.mean.append([])
    #         self.std_dev.append([])
    #         self.var.append([])
    #         self.hist.append([])
    #         self.allan_var.append([])

    async def run(samples):

        # mean = []
        # std_dev = []
        # var = []
        # hist = []
        # allan_var = []

        # for _ in range(config.N_POLS):
        #     mean.append([])
        #     std_dev.append([])
        #     var.append([])
        #     hist.append([])
        #     allan_var.append([])

        # def mean(samples):
        #     return np.mean(samples)
        
        # def std_dev(samples):
        #     return np.std(samples)
        
        # def var(samples):
        #     return np.var(samples)

        # def histogram(samples):
        #     return np.histogram(samples,config.hist_res)

        mean = []
        std_dev = []
        var = []
        hist = []
        allan_dev = []

        for pol in range(len(samples)):
            # mean[pol].append(mean(samples[pol]))
            # std_dev[pol].append(std_dev(samples[pol]))
            # var[pol].append(var(samples[pol]))
            # hist[pol].append(histogram(samples[pol]))
            # allan_var[pol].append(allan_var.allan_variance(samples[pol]))
            
            mean.append(np.mean(samples[pol]))
            std_dev.append(np.std(samples[pol]))
            var.append(np.var(samples[pol]))
            hist.append(np.histogram(samples[pol]))
            allan_dev.append(allan_var.allan_variance(samples[pol]))



        # return self.mean, self.std_dev, self.var, self.hist, self.allan_var
        return mean, std_dev, var, hist, allan_dev