# -*- coding: utf-8 -*-

from itertools import product

from IPython.html.widgets import interact, interactive, fixed
from IPython.display import display
from IPython.html import widgets
import numpy as np
import pandas as pd
from scipy.stats import linregress

from .visualize import plot_scalefree, plot_r2s

__all__ = ['NetworkBuilder']

class NetworkBuilder(object):

    def __init__(self, X=None, correlation=None):
        """Initialize the correlation maker

        :param X:
        :type X:
        :param correlation:
        :type correlation:
        :param abs:
        :type abs:
        :return:
        :rtype:
        """

        self.X = X
        self.correlation = correlation

    def correlate(self, method='spearman', axis=1):
        """Correlate the columns or rows of a dataframe

        The "method" argument is exactly the same as the pandas dataframe
        correlation method.

        Parameters
        ----------
        method : {'spearman', 'pearson', 'kendall'}
            pearson : standard correlation coefficient
            kendall : Kendall Tau correlation coefficient
            spearman : Spearman rank correlation
        axis : {1, 0}, optional
            0 : Calculate correlations between rows
            1 : Calculate correlations between columns

        Returns
        -------
        correlated : pandas.DataFrame
            A (nrows, nrows) or (ncols, ncols) sized matrix of correlations

        """

        if axis == 0:
            X = self.X.T
        else:
            X = self.X

        corr = X.corr(method=method)

        # Replace diagonal self-correlations with 0
        corr.values[np.diag_indices_from(corr)] = 0
        return corr

    def weight(self, correlated, power=2):
        """Exponentiate the correlations to weight them

        Parameters
        ----------
        correlated : pandas.DataFrame
            An (n, m) matrix of correlations
        power : float, optional
            The power to raise the values in the matrix to

        Returns
        -------
        adjacency : pandas.DataFrame
            An (n, m) matrix of correlations
        """
        return correlated.pow(power)

    def node_pair_connectivity(self, adjacency, axis=0):
        """Calculate the sum of all pairwise correlation products.

        In Zhang and Horvath 2005, this is described immediately after
        equation 5 as,

        l_{ij} = \sum_u a_{iu} a_{uj}

        Parameters
        ----------
        adjacency : pandas.DataFrame
            A (nrows, ncols) adjacency correlation matrix
        axis : {0, 1}
            0 : Get (ncols, ncols) between-column pairwise correlations
            1 : Get (nrows, nrows) between-row pairwise correlation matrix

        Returns
        -------
        pairwise_connectivity : pandas.DataFrame
            Square matrix of sum of pairwise products of connectivity
        """
        return adjacency.apply(
            lambda x: (adjacency * x).sum(axis=axis), axis=axis)

    def node_connectivity(self, adjacency, axis=0):
        """Calculate the connectivity of a node by summing all correlations

        Parameters
        ----------
        adjacency : pandas.DataFrame
            A (nrows, ncols) adjacency correlation matrix
        axis : {0, 1}
            0 : Sum over all rows, result is a series with columns as index
            1 : sum over all columns, result is a series with index as index

        Returns
        -------
        connectivity : pandas.Series
            Sum of all connectivity per node
        """
        return adjacency.abs().sum(axis=axis)

    def min_node_connectivity(self, connectivity):
        """For each pair of items, get the minimum

        Parameters
        ----------
        connectivity : pandas.Series
            A (n,) vector of per-node connectivity

        Returns
        -------
        min_pairwise_connectivity : pandas.DataFrame
            A (n, n) dataframe of the minimum connectivity per pair
        """
        n = connectivity.shape[0]
        k_ij = np.empty((n, n), dtype=float)
        for i, a in enumerate(connectivity):
            for j, b in enumerate(connectivity):
                k_ij[i, j] = min(a, b)
        k_ij = pd.DataFrame(k_ij, index=connectivity.index, columns=connectivity.index)
        return k_ij

    def biweight_midcorrelation(self, X):
        median = X.median()
        mad = (X - median).abs().median()
        U = (X - median) / (9 * mad)
        adjacency = np.square(1 - np.square(U)) * ((1 - U.abs()) > 0)
        estimator = (X - median) * adjacency

        bicor_matrix = np.empty((X.shape[1], X.shape[1]), dtype=float)

        for i, ac in enumerate(estimator):
            for j, bc in enumerate(estimator):
                a = estimator[ac]
                b = estimator[bc]

                c = (a * b).sum() / (
                    np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()))
                bicor_matrix[i, j] = c
                bicor_matrix[j, i] = c
        return pd.DataFrame(bicor_matrix, index=X.columns, columns=X.columns)

    def topological_overlap(self, correlated, adjacency, axis=0, signed=True):
        adjacency =
        pairwise_connectivity = self.node_pair_connectivity(adjacency,
                                                            axis=axis)
        connectivity = self.node_connectivity(adjacency, axis=axis)
        min_pairwise_connectivity = self.min_node_connectivity(connectivity)
        modified_adjacency = np.copysign(correlated, adjacency)

        # topological overlap matrix
        if signed:
            tom = (pairwise_connectivity + adjacency).abs() / (
                min_pairwise_connectivity + 1 - adjacency.abs())

        else:
            tom = (pairwise_connectivity.abs() + adjacency.abs())/(
                min_pairwise_connectivity + 1 - adjacency.abs())
        return tom

    def dissimilarity(self, tom):
        """Get topological overlap matrix-based dissimilarity

        dissimilarity = 1 - tom

        Parameters
        ----------
        tom : pandas.DataFrame
            Topological overlap matrix

        Returns
        -------
        dissimilar : pandas.DataFrame
            Dissimilarity matrix
        """
        return 1 - tom

    def pick_soft_threshold(self):
        w = interactive(self._soft_threshold,
                        method=('spearman', 'pearson', 'kendall'),
                  power=widgets.IntSliderWidget(min=0, max=100, step=2,
                                                value=10),
                  signed=True)
        display(w)

    def _soft_threshold(self, method='spearman', power=2, signed=True):
        correlated = self.correlate(method=method)
        adjacency = self.weight(correlated, power=power)
        # tom = builder.topological_overlap(adjacency)
        connectivity = self.node_connectivity(adjacency)
        return plot_scalefree(connectivity)

    def pick_hard_threshold(self):
        w = interactive(self._soft_threshold,
                        method=('spearman', 'pearson', 'kendall'),
                        power=widgets.FloatSliderWidget(min=0, max=.99,
                                                        step=.01, value=0.5),
                        signed=True)
        display(w)

    def _hard_threshold(self, method='spearman', thresh=0.5, signed=True):
        correlated = self.correlate(method=method)

        if not signed:
            correlated = correlated.abs()

        correlated[correlated.abs() < thresh] = 0
        connectivity = self.node_connectivity(correlated)
        return plot_scalefree(connectivity)

    def show_scalefree_r2_different_powers(self, powers=np.arange(2, 202, 2),
                                           ax=None):
        r2s = np.empty(powers.shape)

        for i, power in enumerate(powers):
            correlated = self.correlate(method='spearman')
            adjacency = self.weight(correlated, power=power)
            # tom = builder.topological_overlap(adjacency)
            connectivity = self.node_connectivity(adjacency)

            hist, bin_edges = np.histogram(connectivity, bins=20)

            x = np.log10(bin_edges[:-1])
            y = np.log10(hist)
            finite = np.isfinite(x) & np.isfinite(y)
            slope, intercept, r, p, stderr = linregress(x[finite], y[finite])
            r_squared = np.square(r)
            r2s[i] = r_squared
        plot_r2s(powers, r2s, ax=ax)
        return r2s