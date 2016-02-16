# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 07:16:06 2016

@author: nightrose
"""

def Function_form(    bBandsPeriod,smaPeriod_short,smaPeriod_long, slopePeriod,plot=False):
    from pyalgotrade.stratanalyzer import returns
    from pyalgotrade.stratanalyzer import sharpe
    from pyalgotrade.stratanalyzer import drawdown
    from pyalgotrade.stratanalyzer import trades

    from pyalgotrade import strategy
    from pyalgotrade.strategy import position
    from pyalgotrade import plotter
    from pyalgotrade.tools import yahoofinance
    from pyalgotrade.barfeed import yahoofeed
    from pyalgotrade.technical import bollinger
    from pyalgotrade.stratanalyzer import sharpe
    import BBands_mod
    #import report
    import matplotlib.pyplot as plt 

    # Load the yahoo feed from the CSV file
    feed = yahoofeed.Feed()
    feed.addBarsFromCSV("orcl", "data_daily.csv")

    #def main(plot):
    instrument = "orcl"


    # Evaluate the strategy with the feed's bars.
    myStrategy = BBands_mod.BBands(feed, instrument, bBandsPeriod,smaPeriod_short,smaPeriod_long,slopePeriod)

    if plot:
        plt = plotter.StrategyPlotter(myStrategy, True, True, True)
        plt.getInstrumentSubplot(instrument).addDataSeries("upper", myStrategy.getBollingerBands().getUpperBand())
        plt.getInstrumentSubplot(instrument).addDataSeries("middle", myStrategy.getBollingerBands().getMiddleBand())
        plt.getInstrumentSubplot(instrument).addDataSeries("lower", myStrategy.getBollingerBands().getLowerBand())

    # Attach different analyzers to a strategy before executing it.
    retAnalyzer = returns.Returns()
    myStrategy.attachAnalyzer(retAnalyzer)
    sharpeRatioAnalyzer = sharpe.SharpeRatio()
    myStrategy.attachAnalyzer(sharpeRatioAnalyzer)
    drawDownAnalyzer = drawdown.DrawDown()
    myStrategy.attachAnalyzer(drawDownAnalyzer)
    tradesAnalyzer = trades.Trades()
    myStrategy.attachAnalyzer(tradesAnalyzer)

    # Run the strategy.
    myStrategy.run()

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
    return (myStrategy.getResult(),retAnalyzer.getCumulativeReturns()[-1] * 100,
            sharpeRatioAnalyzer.getSharpeRatio(0.05), drawDownAnalyzer.getMaxDrawDown() * 100)
            
list1=[]
for bBandsPeriod in (30,35,41):
    for smaPeriod_short in (5,10,15,20):
        for smaPeriod_long in (21,30,45):
            for slopePeriod in (30,37,45):
                    Total,C_return,Sharpe,MaxDrawdown=Function_form(bBandsPeriod,smaPeriod_short,smaPeriod_long, 
                                                                    slopePeriod,plot=False)
                    list1.append((bBandsPeriod,smaPeriod_short,smaPeriod_long,slopePeriod,Total,C_return,Sharpe,MaxDrawdown))