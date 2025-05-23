import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def simple_graph(X,Y,title='X vs Y', xlabel="X", ylabel='Y'):
    plt.figure(figsize=(10,10))
    plt.scatter(X,Y,color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    
def scatter(X,Y,title='X vs Y', xlabel="X", ylabel='Y'):
    plt.figure(figsize=(10,10))
    plt.scatter(X,Y,color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    
def histogram(column, title='', xlabel=''):
    plt.hist((column), bins=50)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()