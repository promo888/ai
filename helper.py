import pandas as pd
# import ray
# ray.init(num_cpus=6)
import modin.pandas as pd

import preproc
from technical_indicators_lib import *
#rom talib_test import bb #
import timeit
import pickle
from yahoo_fin import stock_info as si

# instantiate the class
obv = OBV()
rsi = RSI()
bb = BollingerBands()
cci = CCI()
roc = ROC()
macd = MACD()
stoch2 = StochasticKAndD()


def saveDataCsv(df_data, filename, data_foler="data"):
    df_data.to_csv(data_foler + "/" + filename)

def loadDataCsv(filename, data_foler="data"):
    return pd.read_csv(data_foler + "/" + filename)

def getDataFromYahoo(symbol, fromDate="2000-01-01", toDate="2021-31-12"):
    startYahoo = timeit.default_timer()
    web_data = si.get_data(symbol, fromDate, toDate)
    web_data.to_csv("data/%s_%s_%s.csv" % (symbol.replace("^", ""), fromDate, toDate))
    print("Get Yahoo ticker %s took %.2f secs" % (symbol, timeit.default_timer() - startYahoo))
    return web_data


def saveModel(model, filename, models_dir="models"):
    try:
        pickle.dump(model, open(models_dir+"/"+filename, 'wb'))
    except:
        pass


def loadModel(filename, models_dir="models"):
    try:
        return pickle.load(open(models_dir + "/" + filename, 'rb'))
    except:
        pass


def pctChange(fromAmount, toAmount, direction="long"):
    if fromAmount == toAmount:
        return 0
    else:
        if direction == "long" and toAmount > fromAmount:
            return (toAmount/fromAmount - 1) * 100           #advance
        if direction == "long" and toAmount < fromAmount:
            return (fromAmount / toAmount - 1) * -1 * 100    #decline
        if direction == "short" and toAmount < fromAmount:
            return (fromAmount / toAmount - 1) * 100         #advance
        if direction == "short" and fromAmount < toAmount:
            return (toAmount / fromAmount - 1) * -1 * 100    #decline


#todo days declinse/advance, percentAdvanceDecline
def preprocData(ohlcv):
    start = timeit.default_timer()
    try:
        ohlcv = pd.read_csv('preproc_data.csv', index_col=0)
        end = timeit.default_timer()
        print("Preproc data load took %.2f seconds" % (end - start))
        return ohlcv
    except:
        pass



    preproc.ema(ohlcv, 'adjclose', 2)
    preproc.ema(ohlcv, 'adjclose', 3)
    preproc.ema(ohlcv, 'adjclose', 5)
    preproc.ema(ohlcv, 'adjclose', 10)
    preproc.ema(ohlcv, 'adjclose', 20)
    preproc.ema(ohlcv, 'adjclose', 50)
    preproc.ema(ohlcv, 'adjclose', 100)

    preproc.ema(ohlcv, 'high', 2)
    preproc.ema(ohlcv, 'high', 3)
    preproc.ema(ohlcv, 'high', 5)
    preproc.ema(ohlcv, 'high', 10)
    preproc.ema(ohlcv, 'high', 20)
    preproc.ema(ohlcv, 'high', 50)
    preproc.ema(ohlcv, 'high', 100)

    preproc.ema(ohlcv, 'low', 2)
    preproc.ema(ohlcv, 'low', 3)
    preproc.ema(ohlcv, 'low', 5)
    preproc.ema(ohlcv, 'low', 10)
    preproc.ema(ohlcv, 'low', 20)
    preproc.ema(ohlcv, 'low', 50)
    preproc.ema(ohlcv, 'low', 100)

    preproc.isUp(ohlcv, 'adjclose')
    preproc.isUp(ohlcv, 'ema_2_adjclose')
    preproc.isUp(ohlcv, 'ema_3_adjclose')
    preproc.isUp(ohlcv, 'ema_5_adjclose')
    preproc.isUp(ohlcv, 'ema_10_adjclose')
    preproc.isUp(ohlcv, 'ema_20_adjclose')
    preproc.isUp(ohlcv, 'ema_50_adjclose')
    preproc.isUp(ohlcv, 'ema_100_adjclose')

    preproc.isUp(ohlcv, 'ema_2_high')
    preproc.isUp(ohlcv, 'ema_3_high')
    preproc.isUp(ohlcv, 'ema_5_high')
    preproc.isUp(ohlcv, 'ema_10_high')
    preproc.isUp(ohlcv, 'ema_20_high')
    preproc.isUp(ohlcv, 'ema_50_high')
    preproc.isUp(ohlcv, 'ema_100_high')

    preproc.isUp(ohlcv, 'ema_2_low')
    preproc.isUp(ohlcv, 'ema_3_low')
    preproc.isUp(ohlcv, 'ema_5_low')
    preproc.isUp(ohlcv, 'ema_10_low')
    preproc.isUp(ohlcv, 'ema_20_low')
    preproc.isUp(ohlcv, 'ema_50_low')
    preproc.isUp(ohlcv, 'ema_100_low')


    preproc.pctChange1p(ohlcv, 'open')
    preproc.pctChange1p(ohlcv, 'adjclose')
    preproc.pctChange1p(ohlcv, 'low')
    preproc.pctChange1p(ohlcv, 'high')

    preproc.pctChange1p(ohlcv, 'ema_2_high')
    preproc.pctChange1p(ohlcv, 'ema_3_high')
    preproc.pctChange1p(ohlcv, 'ema_5_high')
    preproc.pctChange1p(ohlcv, 'ema_10_high')
    preproc.pctChange1p(ohlcv, 'ema_20_high')
    preproc.pctChange1p(ohlcv, 'ema_50_high')
    preproc.pctChange1p(ohlcv, 'ema_100_high')

    preproc.pctChange1p(ohlcv, 'ema_2_adjclose')
    preproc.pctChange1p(ohlcv, 'ema_3_adjclose')
    preproc.pctChange1p(ohlcv, 'ema_5_adjclose')
    preproc.pctChange1p(ohlcv, 'ema_10_adjclose')
    preproc.pctChange1p(ohlcv, 'ema_20_adjclose')
    preproc.pctChange1p(ohlcv, 'ema_50_adjclose')
    preproc.pctChange1p(ohlcv, 'ema_100_adjclose')

    preproc.pctChange1p(ohlcv, 'ema_2_low')
    preproc.pctChange1p(ohlcv, 'ema_3_low')
    preproc.pctChange1p(ohlcv, 'ema_5_low')
    preproc.pctChange1p(ohlcv, 'ema_10_low')
    preproc.pctChange1p(ohlcv, 'ema_20_low')
    preproc.pctChange1p(ohlcv, 'ema_50_low')
    preproc.pctChange1p(ohlcv, 'ema_100_low')

    preproc.volatility(ohlcv, 'adjclose', 2)
    preproc.volatility(ohlcv, 'adjclose', 3)
    preproc.volatility(ohlcv, 'adjclose', 5)
    preproc.volatility(ohlcv, 'adjclose', 10)
    preproc.volatility(ohlcv, 'adjclose', 20)
    preproc.volatility(ohlcv, 'adjclose', 50)
    preproc.volatility(ohlcv, 'adjclose', 100)

    preproc.volatility(ohlcv, 'high', 2)
    preproc.volatility(ohlcv, 'high', 3)
    preproc.volatility(ohlcv, 'high', 5)
    preproc.volatility(ohlcv, 'high', 10)
    preproc.volatility(ohlcv, 'high', 20)
    preproc.volatility(ohlcv, 'high', 50)
    preproc.volatility(ohlcv, 'high', 100)

    preproc.volatility(ohlcv, 'low', 2)
    preproc.volatility(ohlcv, 'low', 3)
    preproc.volatility(ohlcv, 'low', 5)
    preproc.volatility(ohlcv, 'low', 10)
    preproc.volatility(ohlcv, 'low', 20)
    preproc.volatility(ohlcv, 'low', 50)
    preproc.volatility(ohlcv, 'low', 100)
    preproc.volatility(ohlcv, 'low', 200)
    preproc.volatility(ohlcv, 'ema_2_low', 2)
    preproc.volatility(ohlcv, 'ema_3_low', 3)
    preproc.volatility(ohlcv, 'ema_5_low', 5)
    preproc.volatility(ohlcv, 'ema_10_low', 10)
    preproc.volatility(ohlcv, 'ema_20_low', 20)
    preproc.volatility(ohlcv, 'ema_50_low', 50)
    preproc.volatility(ohlcv, 'ema_100_low', 100)


    # preproc.rsi(ohlcv, "adjclose", 14)
    # preproc.rsi(ohlcv, "high", 14)
    # preproc.rsi(ohlcv, "low", 14)
    # preproc.rsi(ohlcv, "open", 14)

    # preproc.rsi(ohlcv, "adjclose", 10)
    # preproc.rsi(ohlcv, "high", 10)
    # preproc.rsi(ohlcv, "low", 10)
    # preproc.rsi(ohlcv, "adjclose", 20)
    # preproc.rsi(ohlcv, "high", 20)
    # preproc.rsi(ohlcv, "low", 20)
    #
    # preproc.rsi(ohlcv, 'ema_2_adjclose')
    # preproc.rsi(ohlcv, 'ema_3_adjclose')
    # preproc.rsi(ohlcv, 'ema_5_adjclose')
    # preproc.rsi(ohlcv, 'ema_10_adjclose')
    # preproc.rsi(ohlcv, 'ema_20_adjclose')
    # preproc.rsi(ohlcv, 'ema_50_adjclose')
    # preproc.rsi(ohlcv, 'ema_100_adjclose')
    #
    # preproc.rsi(ohlcv, 'ema_2_high')
    # preproc.rsi(ohlcv, 'ema_3_high')
    # preproc.rsi(ohlcv, 'ema_5_high')
    # preproc.rsi(ohlcv, 'ema_10_high')
    # preproc.rsi(ohlcv, 'ema_20_high')
    # preproc.rsi(ohlcv, 'ema_50_high')
    # preproc.rsi(ohlcv, 'ema_100_high')
    #
    preproc.rsi(ohlcv, 'ema_2_low')
    preproc.rsi(ohlcv, 'ema_3_low')
    preproc.rsi(ohlcv, 'ema_5_low')
    preproc.rsi(ohlcv, 'ema_10_low')
    preproc.rsi(ohlcv, 'ema_20_low')
    preproc.rsi(ohlcv, 'ema_50_low')
    preproc.rsi(ohlcv, 'ema_100_low')
    #
    # preproc.rsi(ohlcv, 'ema_2_low', 20)
    #preproc.rsi(ohlcv, 'ema_3_low', 10)
    #preproc.rsi(ohlcv, 'ema_5_low', 10)
    # preproc.rsi(ohlcv, 'ema_10_low', 10)
    # preproc.rsi(ohlcv, 'ema_20_low', 10)
    #preproc.rsi(ohlcv, 'ema_50_low', 5)
    #preproc.rsi(ohlcv, 'ema_100_low', 5)
    # #
    # preproc.rsi(ohlcv, 'ema_2_high', 20)
    # preproc.rsi(ohlcv, 'ema_3_high', 20)
    # preproc.rsi(ohlcv, 'ema_5_high', 20)
    # preproc.rsi(ohlcv, 'ema_10_high', 10)
    # preproc.rsi(ohlcv, 'ema_20_high', 10)
    # preproc.rsi(ohlcv, 'ema_50_high', 5)
    # preproc.rsi(ohlcv, 'ema_100_high', 5)

    #preproc.stoch(ohlcv, 'adjclose', 5) #5+14 SAME ACCURACY,but together less
    preproc.stoch(ohlcv, 'adjclose', 14) #TODO REMOVE CORRELATING FEATURES FOR ACCURATE PREDICTIONS
    preproc.stoch(ohlcv, 'adjclose', 25) #todo Optimize best? + bad together other periods
    preproc.stoch(ohlcv, 'open', 25)
    preproc.stoch(ohlcv, 'high', 25)
    preproc.stoch(ohlcv, 'low', 25)

    # preproc.stoch(ohlcv, 'ema_2_low', 14)
    # preproc.stoch(ohlcv, 'ema_3_low', 14)
    # preproc.stoch(ohlcv, 'ema_5_low', 14)
    # preproc.stoch(ohlcv, 'ema_10_low', 10)
    # preproc.stoch(ohlcv, 'ema_20_low', 10)
    # preproc.stoch(ohlcv, 'ema_50_low', 5)
    # preproc.stoch(ohlcv, 'ema_100_low', 5)

    # preproc.isPeriodHighBack(ohlcv, 3)
    # preproc.isPeriodHighBack(ohlcv, 5)
    # preproc.isPeriodHighBack(ohlcv, 9)
    # preproc.isPeriodHighBack(ohlcv, 14)
    # preproc.isPeriodHighBack(ohlcv, 18)
    # preproc.isPeriodHighBack(ohlcv, 50)
    #
    # preproc.isPeriodHighBack(ohlcv, 3, "low")
    # preproc.isPeriodHighBack(ohlcv, 5, "low")
    # preproc.isPeriodHighBack(ohlcv, 9, "low")
    # preproc.isPeriodHighBack(ohlcv, 14, "low")
    # preproc.isPeriodHighBack(ohlcv, 18, "low")
    # preproc.isPeriodHighBack(ohlcv, 50, "low")
    #
    # preproc.isPeriodHighBack(ohlcv, 3, "adjclose")
    # preproc.isPeriodHighBack(ohlcv, 5, "adjclose")
    # preproc.isPeriodHighBack(ohlcv, 9, "adjclose")
    # preproc.isPeriodHighBack(ohlcv, 14, "adjclose")
    # preproc.isPeriodHighBack(ohlcv, 18, "adjclose")
    # preproc.isPeriodHighBack(ohlcv, 50, "adjclose")

    # preproc.isPeriodLowBack(ohlcv, 3)
    # preproc.isPeriodLowBack(ohlcv, 5)
    # preproc.isPeriodLowBack(ohlcv, 9)
    # preproc.isPeriodLowBack(ohlcv, 14)
    # preproc.isPeriodLowBack(ohlcv, 18)
    # preproc.isPeriodLowBack(ohlcv, 50)
    #
    # preproc.isPeriodLowBack(ohlcv, 3, "adjclose")
    # preproc.isPeriodLowBack(ohlcv, 5, "adjclose")
    # preproc.isPeriodLowBack(ohlcv, 9, "adjclose")
    # preproc.isPeriodLowBack(ohlcv, 14, "adjclose")
    # preproc.isPeriodLowBack(ohlcv, 18, "adjclose")
    # preproc.isPeriodLowBack(ohlcv, 50, "adjclose")
    #
    # preproc.isPeriodLowBack(ohlcv, 3, "open")
    # preproc.isPeriodLowBack(ohlcv, 5, "open")
    # preproc.isPeriodLowBack(ohlcv, 9, "open")
    # preproc.isPeriodLowBack(ohlcv, 14, "open")
    # preproc.isPeriodLowBack(ohlcv, 18, "open")
    # preproc.isPeriodLowBack(ohlcv, 50, "open")

    #preproc.pctChangePriceByPeriod(ohlcv, 2)
    #preproc.pctChangePriceByPeriod(ohlcv, 3)
    #preproc.pctChangePriceByPeriod(ohlcv, 5)


    preproc.pctChangePriceByPeriod(ohlcv, 9)
    preproc.pctChangePriceByPeriod(ohlcv, 18)
    preproc.pctChangePriceByPeriod(ohlcv, 48)
    preproc.pctChangePriceByPeriod(ohlcv, 98)

    # preproc.isPctChangePriceByPeriod(ohlcv, 5, 3) #todo rectified cross tuning for same class sequences
    # preproc.isPctChangePriceByPeriod(ohlcv, 5, 5)
    # preproc.isPctChangePriceByPeriod(ohlcv, 5, 10)
    # preproc.isPctChangePriceByPeriod(ohlcv, 5, 20)
    #preproc.isPctChangePriceByPeriod(ohlcv, 5, 50)
   # preproc.isPctChangePriceByPeriod(ohlcv, 5, 100)
   # preproc.isPctChangePriceByPeriod(ohlcv, 10, 200)

    preproc.isHighLowFromPeriodBack(ohlcv, 0.3, 2)
    preproc.isHighLowFromPeriodBack(ohlcv, 0.5, 2)
    preproc.isHighLowFromPeriodBack(ohlcv, 1, 2)
    preproc.isHighLowFromPeriodBack(ohlcv, 2, 2)
    preproc.isHighLowFromPeriodBack(ohlcv, 3, 2)
    preproc.isHighLowFromPeriodBack(ohlcv, 5, 2)

    preproc.isHighLowFromPeriodBack(ohlcv, 0.3, 3)
    preproc.isHighLowFromPeriodBack(ohlcv, 0.5, 3)
    preproc.isHighLowFromPeriodBack(ohlcv, 1, 3)
    preproc.isHighLowFromPeriodBack(ohlcv, 3, 3)
    preproc.isHighLowFromPeriodBack(ohlcv, 3, 3)
    preproc.isHighLowFromPeriodBack(ohlcv, 5, 3)

    preproc.isHighLowFromPeriodBack(ohlcv, 0.3, 5)
    preproc.isHighLowFromPeriodBack(ohlcv, 0.5, 5)
    preproc.isHighLowFromPeriodBack(ohlcv, 1, 5)
    preproc.isHighLowFromPeriodBack(ohlcv, 2, 5)
    preproc.isHighLowFromPeriodBack(ohlcv, 3, 5)
    preproc.isHighLowFromPeriodBack(ohlcv, 5, 5)
    preproc.isHighLowFromPeriodBack(ohlcv, 10, 10)




    #no impact ? todo crossover with accuracy
    #preproc.stochrsi(ohlcv, 'adjclose', 25)
    #preproc.stochrsi(ohlcv, 'low', 10)
    #preproc.stochrsi(ohlcv, 'high', 10)
    #preproc.stochrsi(ohlcv, 'adjclose', 10)

    #preproc.williams(ohlcv, 'adjclose', 25)
    #preproc.williams(ohlcv, 'adjclose', 50)

    #preproc.ppo(ohlcv, 'adjclose', 20, 10, 5)
    #preproc.ppo(ohlcv, 'adjclose', 10, 5, 3)
    #preproc.ppo(ohlcv, 'adjclose', 50, 20, 10)
    #preproc.ppo(ohlcv, 'low', 10, 5, 3)

    #preproc.pvo(ohlcv, 'adjclose')
    #preproc.pvo(ohlcv, 'low')
    #preproc.pvo(ohlcv, 'adjclose', 10, 5, 3)

    # preproc.tsi(ohlcv, 'adjclose')
    # preproc.tsi(ohlcv, 'low')
    # preproc.tsi(ohlcv, 'high')



    #bad impact
    #preproc.roc(ohlcv, 'adjclose', 10)
    # preproc.roc(ohlcv, 'adjclose', 25)
    #preproc.roc(ohlcv, 'low', 10)
    #preproc.roc(ohlcv, 'high', 5)

    #preproc.kama(ohlcv, 'adjclose', 25)



    #preproc.bbands(ohlcv, 'adjclose', 20)
    #preproc.bb(ohlcv, 'adjclose', 20)



    # Method 1: get the data by sending a dataframe
    #df = obv.get_value_df(ohlcv)
    #df = rsi.get_value_df(ohlcv, 3)
    #df = rsi.get_value_df(ohlcv, 5)
    #df = rsi.get_value_df(ohlcv, 9)
    #df = rsi.get_value_df(ohlcv, 14)
    #df = rsi.get_value_df(ohlcv, 21)


    # Method 2: get the data by sending series values
    # obv_values = obv.get_value_list(df["close"], df["volume"])

    # preproc.bbp(ohlcv, 'adjclose')
    # preproc.bbp(ohlcv, 'high') #todo periodNumHigh
    # preproc.bbp(ohlcv, 'low')

    end = timeit.default_timer()
    print("Preproc data took %.2f seconds" % (end-start))
    ohlcv.to_csv('preproc_data.csv')