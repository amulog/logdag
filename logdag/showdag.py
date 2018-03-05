#!/usr/bin/env python
# coding: utf-8


import sys
import pickle
import networkx as nx

from . import arguments


class LogDAG():

    def __init__(self, args, graph = None):
        self.args = args
        self.graph = graph

    def dump(self):
        fp = arguments.ArgumentManager.dag_filepath(self.args)
        with open(fp, 'wb') as f:
            pickle.dump(self.graph, f)

    def load(self):
        fp = arguments.ArgumentManager.dag_filepath(self.args)
        with open(fp, 'rb') as f:
            self.graph = pickle.load(f)


def empty_dag():
    """nx.DiGraph: Return empty graph."""
    return nx.DiGraph()



