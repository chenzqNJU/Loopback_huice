import numpy as np
from time import time
import pandas as pd
import math
import datetime
import tushare as ts
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.font_manager import FontProperties
from urllib.request import urlretrieve
import os
import myfunc
import imp
import seaborn as sns
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
sns.set_style('white')

imp.reload(myfunc)

Price = myfunc.get_price(N_year=4)


cons = myfunc.get_cons()
capital_base = 100000000  # 起始资金
freq = 'd'  # 策略类型，'d'表示日间策略使用日线回测
dates_pro = myfunc.get_date_list('2017-01-01', '2019-03-15', pro=1)
dates = dates_pro.calendarDate.tolist()
benchmark = 'ZZ500'
r_free = 0.02
N = len(dates)
begin = dates[0]
end = dates[-1]


# refresh_rate = Monthly(1)

class account:
    def __new__(cls, *args, **kwargs):
        cls.date = myfunc.get_date_list() if 'dates' not in globals().keys() else dates
        cls.begin = cls.date[0] if 'begin' not in globals().keys() else begin
        cls.end = cls.date[-1] if 'end' not in globals().keys() else end
        cls.cash = 10000 if 'capital_base' not in globals().keys() else capital_base
        cls.benchmark = 'SHCI' if 'benchmark' not in globals().keys() else benchmark  # 上证综指
        cls.r_free = 2 if 'r_free' not in globals().keys() else r_free * 100  # 上证综指
        return object.__new__(cls, *args, **kwargs)

    def __iter__(self):
        self.days_counter = -1
        return self

    def __next__(self):
        print(self.tradeDate)
        if self.days_counter == len(self.date) - 1:
            self.indicator()
            self.pic()
            print('回测结束')
            raise StopIteration
        else:
            self.days_counter += 1
            self.tradeDate = self.date[self.days_counter]
            self.referencePortfolioValue()
        return self

    def __init__(self):
        self.strategy_name = '策略基准累计收益率'
        self.tradeDate = self.begin
        self.secpos = pd.Series()  # 当前持仓股票index，及持仓量values
        self.Valid_secpos = pd.Series()
        self.price = pd.Series()  # 当前持仓股票index，及当前价格price
        self.if_cost = 0  # 不计算cost
        self.Cost = pd.DataFrame()
        self.SecValue = pd.Series({self.begin: 0})  # 股票价值
        self.PortfolioValue = pd.Series({self.begin: self.cash})  # 总价值
        self.Values = self.PortfolioValue[-1]
        self.days_counter = self.date.index(self.begin)  # 天数从0计数
        # self.bt = pd.DataFrame(columns=['tradeDate', 'cash', 'security_trades', 'security_position', 'portfolio_value',
        #                                 'benchmark_return'],index=range(48))
        self.perf = None

    @property
    def yesterday(self):
        if self.days_counter > 0:
            return (self.date[self.days_counter - 1])
        else:
            return myfunc.get_advanceDate(self.tradeDate)

    def get_positions(self):
        return self.secpos.index.tolist()

    def indicator(self):
        t = {'SHCI': '上证综指', 'HS300': '沪深300', 'ZZ500': '中证500', 'SH50': '上证50'}
        idx = myfunc.get_index(t[self.benchmark], length=1000).rename('基准')
        self.perf = self.PortfolioValue.to_frame(name='策略').join(idx)
        self.net_perf = self.perf.apply(lambda x: x / self.perf.iloc[0, :], axis=1)
        self.perf = self.net_perf * 100 - 100
        self.perf['超额收益'] = self.perf['策略'] - self.perf['基准']
        self.Indicator = dict()
        ############# 年华收益
        ret_year = np.sign(self.perf.iloc[-1, 0]) * abs(self.perf.iloc[-1, 0]) ** (250 / self.days_counter)
        ret_benchmark_year = np.sign(self.perf.iloc[-1, 1]) * abs(self.perf.iloc[-1, 1]) ** (250 / self.days_counter)
        ret_excess_year = ret_year - ret_benchmark_year
        self.Indicator['策略年化收益'] = ret_year
        self.Indicator['基准年化收益'] = ret_benchmark_year
        self.Indicator['超额年华收益'] = ret_excess_year
        ############## 回撤
        self.Indicator['策略回撤'] = self._cal_maxdrawdown('策略')
        self.Indicator['最大回撤'] = self.Indicator['策略回撤'][0]
        self.Indicator['基准回撤'] = self._cal_maxdrawdown('基准')
        ############## 近年来收益
        self.Indicator['2019以来收益'] = self._ret_from_year(year=2019)
        ##############  常见指标
        Indicator = self._get_basic_stat()
        self.Indicator['alpha'] = Indicator[0]
        self.Indicator['beta'] = Indicator[1]
        self.Indicator['年化收益波动率'] = Indicator[2]
        self.Indicator['夏普比率'] = Indicator[3]
        self.Indicator['信息比率'] = Indicator[4]
        self.Indicator['win_loss'] = Indicator[5:]  # 平均涨幅，平均跌幅，胜率

    def _cal_maxdrawdown(self, name):
        values = self.perf[name] + 100
        dd = [values[i:].min() / values[i] - 1 for i in range(len(values))]
        dd = np.array(dd)
        maxdrawdown = dd.min()
        t = dd.argmin()
        maxdrawdown_len = values.iloc[t:].values.argmin()
        maxdrawdown_range = [values.index[t], values.index[t + maxdrawdown_len]]
        return [maxdrawdown, maxdrawdown_len, maxdrawdown_range]  # 回撤、长度、起始日

    def _ret_from_year(self, year=2019):
        values_interval = self.perf[self.perf['策略'].index >= str(year) + '-01-01']
        ret = (values_interval.iloc[-1] + 100) / (values_interval.iloc[0] + 100) * 100 - 100
        ret['超额收益'] = ret['策略'] - ret['基准']
        return ret.tolist()  # 策略、基准、超额收益

    def _get_basic_stat(self):
        '''
         输入 self.perf['cumulative_values']
        '''
        # from sklearn.linear_model import LinearRegression
        values = self.net_perf
        ret_daily = values.pct_change().dropna() * 100
        p = ret_daily['策略']
        wincount = p[p > 0].count()
        winaverage = p[p > 0].sum() / wincount
        losecount = p[p < 0].count()
        loseaverage = p[p < 0].sum() / losecount
        win2lose = abs(winaverage / loseaverage)

        cov = ret_daily.cov().values
        beta = cov[1, 0] / cov[1, 1]
        alpha = (self.Indicator['策略年化收益'] - (self.Indicator['基准年化收益'] - self.r_free) * beta).mean() - self.r_free

        Volatility = np.sqrt(cov[0, 0] * 250)

        SharpRatio = (self.Indicator['策略年化收益'] - self.r_free) / (Volatility)

        sigma_dif = (ret_daily['策略'] - ret_daily['基准']).std() * np.sqrt(250)  # 残差的年华波动

        InformationRatio = (self.Indicator['策略年化收益'] - self.Indicator['基准年化收益']) / sigma_dif

        return [alpha, beta, Volatility, SharpRatio, InformationRatio, winaverage, loseaverage,
                1. * wincount / len(values)]

    def pic(self):
        from matplotlib import ticker as mticker
        from matplotlib.font_manager import FontProperties
        font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
        font_title = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=16)
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(111)

        lns1 = ax1.plot(self.perf.index, self.perf.iloc[:, 0], '-', c='royalblue', label='策略')
        lns2 = ax1.plot(self.perf.index, self.perf.iloc[:, 1], '-', c='grey', label='基准')
        lns3 = ax1.plot(self.perf.index, self.perf.iloc[:, 2], '-', c='orange', label='超额收益')

        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax1.legend(lns, labs, prop=font, loc=2, fontsize=12)

        # ax1.set_xlim(-10,250)
        ax1.set_ylabel(u'累计收益率', fontproperties=font, fontsize=16)
        ax1.set_xlabel(u'日期', fontproperties=font, fontsize=16)
        ax1.set_yticklabels([str(x) + '0%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14)
        ax1.set_title(self.strategy_name, fontproperties=font_title)

        key = ['策略年化收益', '基准年化收益', 'alpha', 'beta', '夏普比率', '信息比率', '年化收益波动率', '最大回撤']
        subtitle = '单位：%   ' + '  '.join(['{}:{:.2f}'.format(i, a.Indicator[i]) for i in key])
        plt.suptitle(subtitle, fontproperties=font, fontdict={'size': '20', 'color': 'dimgrey'}, x=0.5, y=0.96)

        ax1.grid()
        plt.savefig('e:\\yk\\pic\\' + self.strategy_name + '.png')
        plt.show()

    def valid_secpos(self):
        self.secpos = self.secpos[self.secpos > 0]

    def cost(self):
        if (self.secpos.empty) & self.Cost.empty:
            return 0
        elif (not self.secpos.empty) & self.Cost.empty:
            self.Cost = self.secpos.to_frame(name='secpos').join(self.price.rename('cost'))
            self.Cost['Total_Cost'] = self.Cost.secpos * self.Cost.cost
        else:
            self.Cost = self.secpos.to_frame(name='a').join(self.Cost).drop('a', axis=1).fillna(0)
            temp = self.secpos - self.Cost['secpos']
            # temp[temp.isnull()] = self.secpos.loc[temp[temp.isnull()].index]
            temp = temp[temp > 0]
            self.Cost.loc[temp.index, 'Total_Cost'] += temp * self.price[temp.index]
            self.Cost.loc[temp.index, 'secpos'] = self.secpos[temp.index]
            self.Cost.cost = self.Cost.Total_Cost / self.Cost.secpos

    def referencePrice(self):  # 持仓股票的价格
        self.valid_secpos()
        self.price = Price.loc[self.tradeDate, self.secpos.index]
        return (self.price)

    def referenceSecValue(self):  # 持仓证券的价值
        self.referencePrice()
        if self.if_cost == 1: self.cost()
        self.SecValue[self.tradeDate] = (self.secpos * self.price).sum()
        return (self.SecValue)

    def referencePortfolioValue(self):
        self.referenceSecValue()
        self.PortfolioValue[self.tradeDate] = self.cash + self.SecValue.loc[self.tradeDate]
        self.Values = self.PortfolioValue[-1]
        return self.PortfolioValue

    def order_pct(self, *args, cost=0):  # 1 对单个股票，交易num数量（'000001.XSHE ',100）
        if isinstance(args[0], str):  # 单个股票
            args = pd.Series(1 if len(args) == 1 else args[1], index=[args[0]])
        else:
            args = args[0]
        args = args * self.Values
        args = args / 100 // Price.loc[self.tradeDate, args.index] * 100
        cash = self.cash - (args * Price.loc[self.tradeDate, args.index]).sum()
        if cash < 0: raise NameError('error:cash<0')
        self.secpos = self.secpos.append(args)
        self.secpos = self.secpos.groupby(self.secpos.index).sum()
        self.cash = cash
        self.referencePortfolioValue()

    def order_pct_to(self, *args):  # 约定：secpos->args 对单个股票，交易到num百分比
        if isinstance(args[0], str):  # 单个股票、pct
            args = pd.Series(args[1], index=[args[0]])
        else:
            args = args[0]
        stk_num = self.Values * args / 100 // Price.loc[self.tradeDate, args.index] * 100  # stk = args.index
        chg = stk_num.sub(self.secpos, axis=0, fill_value=0)
        self.secpos = stk_num
        chg = chg[chg != 0]
        self.cash -= (chg * Price.loc[a.tradeDate, chg.index]).sum()
        self.referencePortfolioValue()

    def order_to(self, *args):  # 约定：args in secpos 三种输入 1,order('0000001',0) 2,order(['01','02'],0) 3,order(Series)
        if isinstance(args[0], str):  # 单个股票、pct
            args = pd.Series(args[1], index=args[0])
        elif isinstance(args[0], list):
            args = pd.Series(args[1], index=args[0])
        else:
            args = args[0]
        if args.empty: return
        chg = args.sub(self.secpos, axis=0).dropna()
        self.secpos[args.index] = args.values
        self.cash -= (chg * Price.loc[a.tradeDate, chg.index]).sum()
        self.referencePortfolioValue()

    def order(self, *args, cost=0):  # 1 对单个股票，交易num数量（'000001.XSHE ',100）
        if isinstance(args[0], str):  # 单个股票
            args = pd.Series(100 if len(args) == 1 else args[1], index=[args[0]])
        else:
            args = args[0]
        cash = self.cash - (args * Price.loc[self.tradeDate, args.index]).sum()
        if cash < 0: raise NameError('error:cash<0')
        self.secpos = self.secpos.append(args)
        self.secpos = self.secpos.groupby(self.secpos.index).sum()
        self.cash = cash
        self.referencePortfolioValue()


def trade(buy_list):
    buy_list = [i for i in buy_list if i in Price.columns]
    position = a.get_positions()
    stk = list(set(position) - set(buy_list))
    a.order_to(stk, 0)

    if len(buy_list) == 0: return
    # weight = 1 / len(buy_list)
    marketvelue = MV.loc[a.tradeDate, buy_list]
    weight = marketvelue / marketvelue.sum()
    a.order_pct_to(weight)
    if a.cash < 0: print('wrong');raise NameError('error:cash<0')


############################# 每个月第一天
month_1 = dates_pro.isMonthEnd.shift(1).fillna(1)
refresh_date = dates_pro[month_1 == 1].calendarDate.tolist()
#############################  10 分类 1 2 3  4 5 6 7  8 9 10

var = ['PE', ]
a = account()
a.strategy_name = '中证500剔除PE较高10%股票'
for x in a:
    if a.yesterday in refresh_date:
        universe = cons.loc[a.tradeDate]
        t1 = PB.loc[a.yesterday, universe]
        t1 = t1.to_frame(name='PB')
        t2 = t1.join(myfunc.quantile(t1.PB), rsuffix='_rank')
        buy_list = t2[t2.PB_rank < 10].index.tolist()
        trade(buy_list)

