"""
Read in custom csv files
"""
import pandas as pd
from zipline.gens.utils import hash_args
from zipline.sources.data_source import DataSource
import datetime
import csv
import numpy as np
from zipline.algorithm import TradingAlgorithm
from pandas.tseries.tools import to_datetime
import matplotlib.pyplot as plt
%matplotlib inline

def get_time(time_str):
    time_array = map(int, time_str.split(":"))
    assert len(time_array) == 2
    assert time_array[0] < 24 and time_array[1] < 61
    return datetime.time(time_array[0], time_array[1])


def gen_ts(date, time):
    return pd.Timestamp(datetime.datetime.combine(date, time))


class DatasourceCSVohlc(DataSource):
    """ expects dictReader for a csv file
     with the following columns (no header)
    symbol,dt, open, high, low, close, volume
    separated by comma,
    dt expected in ISO format and order does not matter"""
    def __init__(self, filename, **kwargs):
        self.filename = filename
        # Unpack config dictionary with default values.
        if 'symbols' in kwargs:
            self.sids = kwargs.get('symbols')
        else:
            self.sids = None
        self.tz_in = kwargs.get('tz_in', "US/Eastern")
        self.start = pd.Timestamp(to_datetime(kwargs.get('start'))).tz_localize('utc')
        self.end = pd.Timestamp(to_datetime(kwargs.get('end'))).tz_localize('utc')
        self._raw_data = None
        self.arg_string = hash_args(filename, **kwargs)

    @property
    def instance_hash(self):
        return self.arg_string

    def raw_data_gen(self):
        previous_ts = None
        with open(self.filename, 'rb') as csvfile:
            self.data = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in self.data:
                ts = pd.Timestamp(to_datetime(row[1])).tz_localize('utc')
                if ts < self.start or ts > self.end:
                    continue
                volumes = {}
                price_volumes = {}
                sid = row[0]
                if self.sids is None or sid in self.sids:
                    if sid not in volumes:
                        volumes[sid] = 0
                        price_volumes[sid] = 0
                        event = {"sid": sid, "type": "TRADE", "symbol": sid}
                        cols = ["open", "high", "low", "close"]
                        event["dt"] = ts
                        event["price"] = float(row[5])
                        event["volume"] = row[6]
                        volumes[sid] += float(event["volume"])
                        price_volumes[sid] += event["price"] * float(event["volume"])
                        event["vwap"] = price_volumes[sid] / volumes[sid]
                        event["open"] = row[2]
                        event["high"] = row[3]
                        event["low"] = row[4]
                        event["close"] = row[5]
                        yield event

    @property
    def raw_data(self):
        if not self._raw_data:
            self._raw_data = self.raw_data_gen()
        return self._raw_data

    @property
    def mapping(self):
        return {
            'sid': (lambda x: x, 'sid'),
            'dt': (lambda x: x, 'dt'),
            'open': (float, 'open'),
            'high': (float, 'high'),
            'low': (float, 'low'),
            'close': (float, 'close'),
            'price': (float, 'price'),
            'volume': (int, 'volume'),
            'vwap': (lambda x: x, 'vwap')
        }


from zipline.api import order_target, record, symbol, history, add_history

class BuyCubist(TradingAlgorithm):  # inherit from TradingAlgorithm
    """This is the simplest possible algorithm that does nothing but
    buy 1 share on each event.
    """
    def initialize(self):
        pass
        #add_history(100, '1d', 'price')
        #add_history(300, '1d', 'price')
        #context.i = 0
#        set_universe(universe.DollarVolumeUniverse(90.0, 90.1))
        
    def handle_data(context, data):  # overload handle_data() method
        context.order('CUBS', 1)  # order SID (=0) and amount (=1 shares)
#        print(data['CUBS'].high)
        price_history = history(bar_count=20, frequency='1d', field='price')
    
if __name__ == "__main__":
    source = DatasourceCSVohlc('/home/nightrose/Documents/Github/try/xyz/data_daily.csv', symbols='CUBS', 
                               start='2009-01-10', end='2009-12-31')
    simple_algo = BuyCubist()
    results = simple_algo.run(source)
    results.portfolio_value.plot()
#    print(source)
