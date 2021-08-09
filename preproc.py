import numpy as np
from technical_indicators_lib import *
from ta.momentum import rsi as RSI, stoch as Stoch, roc as Roc, \
    kama as Kama, stochrsi as Stochrsi, williams_r as Williams, \
    ppo as Ppo, pvo as Pvo, tsi as Tsi
from ta.volatility import BollingerBands as BBANDS

#import tensorflow

#todo isUpDownByPeriod
def isUp(df_data, field):
    weight = 1 #True #todo assert in rules
    # if "is" in field.lower() and "low" in field.lower():
    #     weight = 1
    #if "isUp_" in field.lower() and "high" in field.lower():
    #    weight = 2  #increment high rule - to differentiate sums for directions - long and shorts
    df_data['diff_' + '%s' % field] = df_data[field].pct_change() > 0 #1 True #diff(1) > 0 #-0.5 #0.333 #0 #tmp fields
    df_data.dropna(inplace=True)
    df_data['isUp_' + '%s' % field] = [weight if x > 0 else 0 for x in df_data['diff_' + '%s' % field]]
    #df_data.drop('diff_' + '%s' % field, inplace=True) #dropped later -check if required
    #df_data.dropna(inplace=True)
    return df_data

def pctChange1p(df_data, field):
    df_data['pctChange_%s' % field] = df_data[field].pct_change() * 100
    return df_data

#todo -HighLow%FromCloseByPeriod
#todo - Done - shiftBack(-periods) to get forward/future PeriodMinMax
#todo forward pct for predictions/actors + training/scroring quorum -> voting classifers/nns/rules
def pctChangePriceByPeriod(df_data, period): #todo to continue
    period_pct_change_high = []
    period_pct_change_low = []
    for x in range(df_data.shape[0]):
        try:
            period_start = x + 1 - period
            period_end = x + 1
            period_pct_high = max(df_data.iloc[x - period: x + 1]["high"])
            period_pct_low = min(df_data.iloc[x - period: x + 1]["low"])
            period_change_pct_high = (period_pct_high / df_data.iloc[x + 1 - period]["adjclose"] - 1) * 100
            period_change_pct_low = (1 - period_pct_low / df_data.iloc[x + 1 - period]["adjclose"]) * 100
            period_pct_change_high.append(period_change_pct_high)
            period_pct_change_low.append(period_change_pct_low)
        except:
            period_pct_change_high.append(None)
            period_pct_change_low.append(None)
    df_data['pctChange_high_%speriods' % (period)] = period_pct_change_high
    df_data['pctChange_low_%speriods' % (period)] = period_pct_change_low
    return df_data

def isHighLowFromPeriodBack(df_data, pct, period): #todo to continue - duplicate above?
    period_pct_change_high = []
    period_pct_change_low = []
    for x in range(df_data.shape[0]):
        try:
            period_start = x + 1 - period
            period_end = x + 1
            period_pct_high = max(df_data.iloc[x - period: period_end]["high"])
            period_pct_low = min(df_data.iloc[x - period: period_end]["low"])
            period_change_pct_high = (period_pct_high / df_data.iloc[period_start]["adjclose"] - 1) * 100
            period_change_pct_low = (1 - period_pct_low / df_data.iloc[period_start]["adjclose"]) * 100
            period_pct_change_high.append(period_change_pct_high)
            period_pct_change_low.append(period_change_pct_low)
        except:
            period_pct_change_high.append(None)
            period_pct_change_low.append(None)
    df_data['%spctChange-isHigh_%speriods_back' % (pct, period)] = [1 if (not x is None and x >= pct) else 0 for x in period_pct_change_high]
    df_data['%spctChange-isLow_%speriods_back' % (pct, period)] = [1 if (not x is None and x >= pct) else 0 for x in period_pct_change_low]
    return df_data

def isHighLowToPeriodForward(df_data, pct, period): #todo to continue
    period_pct_change_high = []
    period_pct_change_low = []
    for x in range(df_data.shape[0]):
        try:
            period_start = x
            period_end = x + period + 1
            period_pct_high = max(df_data.iloc[x - period: x + 1]["high"])
            period_pct_low = min(df_data.iloc[x - period: x + 1]["low"])
            period_change_pct_high = (period_pct_high / df_data.iloc[x + 1 - period]["adjclose"] - 1) * 100
            period_change_pct_low = (1 - period_pct_low / df_data.iloc[x + 1 - period]["adjclose"]) * 100
            period_pct_change_high.append(period_change_pct_high)
            period_pct_change_low.append(period_change_pct_low)
        except:
            period_pct_change_high.append(None)
            period_pct_change_low.append(None)
    df_data['%spctChange-high_%speriods_forward' % (pct, period)] = [1 if (not x is None and x >= pct) else 0 for x in period_pct_change_high]
    df_data['%spctChange-low_%speriods_forward' % (pct, period)] = [1 if (not x is None and x >= pct) else 0 for x in period_pct_change_low]
    return df_data



def isPeriodHighFromBack(df_data, period, field="high"):
    period_high_arr = []
    for x in range(df_data.shape[0]):
        period_high = df_data.iloc[x + 1 - period:x + 1][field]
        if len(period_high) == period:
            if df_data.iloc[x][field] >= max(period_high):
                period_high_arr.append(1)
            else:
                period_high_arr.append(0)
        else:
            period_high_arr.append(None)
    df_data['is_period%s_high_%s' % (period, field)] = period_high_arr
    return df_data


def isPeriodLowBack(df_data, period, field="low"):
    period_low_arr = []
    for x in range(df_data.shape[0]):
        period_low = df_data.iloc[x + 1 - period:x + 1][field]
        if len(period_low) == period:
            if df_data.iloc[x][field] <= min(period_low):
                period_low_arr.append(1)
            else:
                period_low_arr.append(0)
        else:
            period_low_arr.append(None)
    df_data['is_period%s_low_%s' % (period, field)] = period_low_arr
    return df_data


#https://www.datacamp.com/community/tutorials/moving-averages-in-pandas?utm_source=adwords_ppc&utm_campaignid=898687156&utm_adgroupid=48947256715&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=332602034343&utm_targetid=aud-517318241987:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=1007993&gclid=Cj0KCQjws536BRDTARIsANeUZ59kaUs6kMtN2WWmsec3sgBow0NSm_wQjJ9S4-LPxOmgll2b4mMpn5caAg6iEALw_wcB
def sma(df_data, field, ma_period): #data asc - oldest first
    df_data['sma_{}_{}'.format(ma_period, field)] = df_data[field].rolling(window=ma_period).mean()
    return df_data

def ema(df_data, field, ma_period): #data asc - oldest first
    df_data['ema_%s_%s' % (ma_period, field)] = df_data[field].rolling(window=ma_period).mean()
    return df_data

def volatility(df_data, field, period): #data asc - oldest first
    df_data['volatility_{}_{}'.format(period, field)] = df_data[field].rolling(window=period).std() * np.sqrt(period)
    return df_data

def rsi(df_data, field="close", period=14):
    df_data['rsi_{}_{}'.format(period, field)] = RSI(df_data[field], period)
    #rsi = RSI(price['adjclose']) #(close) #, timeperiod=14)
    return df_data

###########
def stoch(df_data, field="close", period=14):
    df_data['stoch_{}_{}'.format(period, field)] = Stoch(df_data["high"], df_data["low"], df_data[field], window=period)
    return df_data

def roc(df_data, field="close", period=12):
    df_data['roc_{}_{}'.format(period, field)] = Roc(df_data[field], window=period)
    return df_data
def kama(df_data, field="close", period=10):
    df_data['kama_{}_{}'.format(period, field)] = Kama(df_data[field], window=period)
    return df_data

def stochrsi(df_data, field="close", period=14):
    df_data['stochrsi_{}_{}'.format(period, field)] = Stochrsi(df_data[field], window=period)
    return df_data

def williams(df_data, field="close", period=14):
    df_data['williams_{}_{}'.format(period, field)] = Williams(df_data["high"], df_data["low"], df_data[field], lbp=period)
    return df_data

def ppo(df_data, field="close", slow_period=26, fast_period=12, signal_period=9):
    df_data['ppo_{}_{}_{}_{}'.format(slow_period, fast_period, signal_period, field)] = Ppo(df_data[field], window_slow=slow_period, window_fast=fast_period, window_sign=signal_period)
    return df_data

def pvo(df_data, field="close", slow_period=26, fast_period=12, signal_period=9):
    df_data['pvo_{}_{}_{}_{}'.format(slow_period, fast_period, signal_period, field)] = Pvo(df_data[field], window_slow=slow_period, window_fast=fast_period, window_sign=signal_period)
    return df_data

def tsi(df_data, field="close", slow_period=26, fast_period=12, signal_period=9):
    df_data['tsi_{}_{}_{}'.format(slow_period, fast_period, signal_period, field)] = Tsi(df_data[field], window_slow=slow_period, window_fast=fast_period)
    return df_data


def bb(df_data, field="close", period=20):
    up, mid, low = BBANDS(df_data[field], window=period)
    bbi = (df_data[field] - low) / (up - low)
    df_data['bbands_{}_{}'.format(period, field)] = bbi
    return df_data #bbi

#Selected Features: ['Open', 'Close', 'isUp_Close', 'sma_2_Close', 'sma_2_High', 'sma_3_High', 'sma_2_Low', 'sma_3_Low', 'ema_2_Close']
# def bbp(df_data, field, period=20, nbdevup=2, nbdevdn=2):
#     up, mid, low = BBANDS(df_data[field], timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0)
#     bbp = (price[field] - low) / (up - low)
#     df_data['BB_{}_{}'.format(period, field)] = bbp
#     #return bbp