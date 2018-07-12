#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common utilities functions

@author: Teddy.Ma
"""

import numpy as np
import operator as op
import pandas as pd
import pydotplus as dot
from IPython.display import Image

# return a sorted dictionary item list
def dicSorted(dicItems, desc = True):
    return sorted(dicItems, key=op.itemgetter(1), reverse=desc)

# load a numpy array from txt file
def loadTable(file):
    return np.array(pd.read_table(file,header=None))


def createDiGraph():
    g = dot.Dot();
    g.set_type('digraph')
    return g

def createNode(name, shape=None, graph=None, label=None, style=None, fillcolor=None):
    node = dot.Node(name)
    if shape is not None: node.set("shape", shape)
    if graph is not None: graph.add_node(node)
    if label is not None: node.set('label', label)
    if style is not None: node.set('style', style)
    if fillcolor is not None: node.set('fillcolor', fillcolor)
    return node

def createEdge(src, dst, graph=None, label=None):
    edge = dot.Edge(src, dst)
    if graph is not None: graph.add_edge(edge)
    if label is not None: edge.set('label', label)
    return edge

def createImage(graph):
    return Image(graph.create_png())

