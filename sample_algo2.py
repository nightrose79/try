# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:57:10 2016

@author: nightrose
"""

import zipline.api import load_from_yahoo, TradingAlgorithm, algo, analyze

# Load price data from yahoo.
data = load_from_yahoo(stocks=['AAPL'], indexes={}, start=start,end=end)

# Create and run the algorithm.
algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data,
                        identifiers=['AAPL'])
results = algo.run(data)

analyze(results=results)