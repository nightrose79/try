from pyalgotrade import strategy
#from pyalgotrade import plotter
#from pyalgotrade.tools import yahoofinance
#from pyalgotrade.barfeed import yahoofeed
from pyalgotrade.technical import bollinger
#from pyalgotrade.stratanalyzer import sharpe


class BBands(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, bBandsPeriod):
        strategy.BacktestingStrategy.__init__(self, feed)
        self.__instrument = instrument
        self.__bbands = bollinger.BollingerBands(feed[instrument].getCloseDataSeries(), bBandsPeriod, 2)

    def getBollingerBands(self):
        return self.__bbands

    def onBars(self, bars):
        lower = self.__bbands.getLowerBand()[-1]
        upper = self.__bbands.getUpperBand()[-1]
        if lower is None:
            return

        shares = self.getBroker().getShares(self.__instrument)
        bar = bars[self.__instrument]
        if shares == 0 and bar.getClose() < lower:
            sharesToBuy = int(self.getBroker().getCash(False) / bar.getClose())
            self.marketOrder(self.__instrument, sharesToBuy)
        elif shares > 0 and bar.getClose() > upper:
            self.marketOrder(self.__instrument, -1*shares)