#!/usr/bin/env python3

import numpy as np
import scipy.stats

def clamped_variance(a, b, s):
    d = scipy.stats.norm(scale=s)
    v = np.square(s)
    return (
        d.cdf(a) * np.square(a)
        + d.sf(b) * np.square(b)
        + (d.cdf(b) - d.cdf(a)) * v
        + d.pdf(a) * a * v
        - d.pdf(b) * b * v
    )

a = -1
b = 1
scales = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625]

for s in scales:
    print(np.sqrt(clamped_variance(a, b, s)))
