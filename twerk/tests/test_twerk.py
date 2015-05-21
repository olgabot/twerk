#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_twerk
----------------------------------

Tests for `twerk` module.
"""

import pandas.util.testing as pdt

class TestNetworkBuilder(object):

    def test_init(self, X, correlation):
        from twerk import NetworkBuilder
        nb = NetworkBuilder(X, correlation)

        pdt.assert_frame_equal(nb.X, X)
        pdt.assert_equal(nb.correlation, correlation)

    def test_correlate(self):
        pass

    def test_weight(self):
        pass

    def test_node_pair_connectivity(self):
        pass

    def test_node_connectivity(self):
        pass

    def test_min_node_connectivity(self):
        pass

    def test_biweight_midcorrelation(self):
        pass

    def test_topological_overlap(self):
        pass

    def test_dissimilarity(self):
        pass

    def test_soft_threshold(self):
        pass

    def test_hard_threshold(self):
        pass

    def test_show_scalefree_r2_different_powers(self):
        pass