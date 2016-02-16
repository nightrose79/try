# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 23:15:11 2016

@author: nightrose
"""
from pyalgotrade.stratanalyzer import returns
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.stratanalyzer import drawdown
from pyalgotrade.stratanalyzer import trades

from pyalgotrade import strategy
from pyalgotrade import plotter
from pyalgotrade.tools import yahoofinance
from pyalgotrade.barfeed import yahoofeed
from pyalgotrade.technical import bollinger
from pyalgotrade.stratanalyzer import sharpe
import matplotlib.pyplot as plt 

def reporting(myStrtegy,plot):
   
    print "Final portfolio value: $%.2f" % myStrategy.getResult()
    print "Cumulative returns: %.2f %%" % (retAnalyzer.getCumulativeReturns()[-1] * 100)
    print "Sharpe ratio: %.2f" % (sharpeRatioAnalyzer.getSharpeRatio(0.05))
    print "Max. drawdown: %.2f %%" % (drawDownAnalyzer.getMaxDrawDown() * 100)
    print "Longest drawdown duration: %s" % (drawDownAnalyzer.getLongestDrawDownDuration())
    
    print
    print "Total trades: %d" % (tradesAnalyzer.getCount())
    if tradesAnalyzer.getCount() > 0:
        profits = tradesAnalyzer.getAll()
        print "Avg. profit: $%2.f" % (profits.mean())
        print "Profits std. dev.: $%2.f" % (profits.std())
        print "Max. profit: $%2.f" % (profits.max())
        print "Min. profit: $%2.f" % (profits.min())
        returns = tradesAnalyzer.getAllReturns()
        print "Avg. return: %2.f %%" % (returns.mean() * 100)
        print "Returns std. dev.: %2.f %%" % (returns.std() * 100)
        print "Max. return: %2.f %%" % (returns.max() * 100)
        print "Min. return: %2.f %%" % (returns.min() * 100)
    
    print
    print "Profitable trades: %d" % (tradesAnalyzer.getProfitableCount())
    if tradesAnalyzer.getProfitableCount() > 0:
        profits = tradesAnalyzer.getProfits()
        print "Avg. profit: $%2.f" % (profits.mean())
        print "Profits std. dev.: $%2.f" % (profits.std())
        print "Max. profit: $%2.f" % (profits.max())
        print "Min. profit: $%2.f" % (profits.min())
        returns = tradesAnalyzer.getPositiveReturns()
        print "Avg. return: %2.f %%" % (returns.mean() * 100)
        print "Returns std. dev.: %2.f %%" % (returns.std() * 100)
        print "Max. return: %2.f %%" % (returns.max() * 100)
        print "Min. return: %2.f %%" % (returns.min() * 100)
    
    print
    print "Unprofitable trades: %d" % (tradesAnalyzer.getUnprofitableCount())
    if tradesAnalyzer.getUnprofitableCount() > 0:
        losses = tradesAnalyzer.getLosses()
        print "Avg. loss: $%2.f" % (losses.mean())
        print "Losses std. dev.: $%2.f" % (losses.std())
        print "Max. loss: $%2.f" % (losses.min())
        print "Min. loss: $%2.f" % (losses.max())
        returns = tradesAnalyzer.getNegativeReturns()
        print "Avg. return: %2.f %%" % (returns.mean() * 100)
        print "Returns std. dev.: %2.f %%" % (returns.std() * 100)
        print "Max. return: %2.f %%" % (returns.max() * 100)
        print "Min. return: %2.f %%" % (returns.min() * 100)
        
    if plot:
        plt.plot()