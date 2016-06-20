from __future__ import division
from pyalgotrade import strategy,broker
from pyalgotrade.strategy import position
from pyalgotrade.technical import bollinger,linreg,ma,cross
#from talib import LINEARREG_SLOPE
import pandas as pd 
import numpy as np


class BBands(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, bBandsPeriod,smaPeriod_short,smaPeriod_long,slope_period):
        strategy.BacktestingStrategy.__init__(self, feed)
        self.__instrument = instrument
        self.__prices = feed[instrument].getPriceDataSeries()
        self.__sma_short = ma.SMA(self.__prices, smaPeriod_short)
        self.__sma_long = ma.SMA(self.__prices, smaPeriod_long)   
        self.__bbands = bollinger.BollingerBands(feed[instrument].getCloseDataSeries(), bBandsPeriod, 2)
        self.__slope = linreg.Slope(self.__prices,slope_period) 
        self.__middle_slope= linreg.Slope(self.__bbands.getMiddleBand(),slope_period)
        self.__longPos = None
        self.__shortPos = None
        self.slope_period=slope_period 


    def getBollingerBands(self):
        return self.__bbands
        
    def getSMA(self):
        return self.__sma
             
    def getSlope(self):
        return self.__slope

    def getMidSlope(self):
        return self.__middle_slope
        
    def onEnterCanceled(self, position):
        if self.__longPos == position:
            self.__longPos = None
        elif self.__shortPos == position:
            self.__shortPos = None
        else:
            assert(False)

    def onExitOk(self, position):
        if self.__longPos == position:
            self.__longPos = None
        elif self.__shortPos == position:
            self.__shortPos = None
        else:
            assert(False)

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        position.exitMarket()
        

        
    def onBars(self, bars):
        bar = bars[self.__instrument]

        
        if self.__longPos is not None:
            if self.exitLongSignal(bar):
                self.__longPos.exitMarket()
        elif self.__shortPos is not None:
            if self.exitShortSignal(bar):
                self.__shortPos.exitMarket()
        else:
            if self.enterLongSignal(bar):# and MACD>0:#self.enterLongSignal(bar):
                shares = int(self.getBroker().getCash() * 0.9 / bars[self.__instrument].getPrice())
                self.__longPos = self.enterLong(self.__instrument, shares)#, True)
            elif self.enterShortSignal():
                shares = int(self.getBroker().getCash() * 0.9 / bars[self.__instrument].getPrice())
                self.__shortPos = self.enterShort(self.__instrument, shares)#, True)
                
    def enterLongSignal(self, bar):
        lower = self.__bbands.getLowerBand()[-1]    
        if lower is None:
            return
        else: 
            if bar.getClose() < lower and self.__middle_slope[-1]> 0:
                print('middle slope is',self.__middle_slope[-1],'buy?',bar.getClose() < lower and self.__middle_slope[-1]> 0)
            return bar.getClose() < lower and self.__middle_slope[-1]> 0


    def exitLongSignal(self,bar):
        shares = self.getBroker().getShares(self.__instrument)   
        Cash=self.getBroker().getCash(False)           
        price=(1000000-Cash)/float(shares)
        upper = self.__bbands.getUpperBand()[-1]
        return bar.getClose() > upper or bar.getClose() <0.95*price
        
    def enterShortSignal(self):
#        for i in range(3):
#            print('MA slope',self.__middle_slope[-1])
        if cross.cross_above(self.__prices, self.__bbands.getUpperBand()) > 0 and self.__middle_slope[-1] <0:
            print('middle slope is',self.__middle_slope[-1],'sell?',cross.cross_above(self.__prices, self.__bbands.getUpperBand()) > 0 and self.__middle_slope[-1] <0)
        return cross.cross_above(self.__prices, self.__bbands.getUpperBand()) > 0 and self.__middle_slope[-1] <0

    def exitShortSignal(self,bar):
        shares = self.getBroker().getShares(self.__instrument)   
        Cash=self.getBroker().getCash(False)           
        price=(1000000-Cash)/float(shares)
        return cross.cross_above(self.__prices, self.__bbands.getLowerBand()) > 0 or bar.getClose()>1.05*price