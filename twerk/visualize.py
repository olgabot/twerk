
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_scalefree(connectivity, ax=None):
    if ax is None:
        ax = plt.gca()

    hist, bin_edges = np.histogram(connectivity, bins=20)

    x = np.log10(bin_edges[:-1])
    y = np.log10(hist)
    ax.plot(x, y, 'o')

    finite = np.isfinite(x) & np.isfinite(y)
    slope, intercept, r, p, stderr = linregress(x[finite], y[finite])
    r2 = np.square(r)
    r2
    xticks = ax.get_xticks()
    y = slope*xticks + intercept
    ax.plot(xticks, y, label='$R^2 = {:g}$'.format(r2))
    ax.set_xlabel('$\log_{10}$ connectivity')
    ax.set_ylabel('$\log_{10}$ Pr(connectivity)')
    ax.legend()

    return slope, intercept, r, p, stderr

def plot_r2s(powers, r2s, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(powers, r2s)
    ax.set_xlabel('$\\beta$ exponent weight of correlation')
    ax.set_ylabel('$R^2$ of fit to linear scale-free topology')