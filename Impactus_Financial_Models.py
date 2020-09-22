#Funções:
# PresentValue(list, rate): calcula o valor presente de uma lista de fluxos 'list'
# a partir de uma taxa de desconto 'rate'

#MA(nPeriods, dfname, origin): calcula uma média móvel simples
#   'nPeriods': número de períodos da média móvel. ex: 100
#   'dfname': nome do dataframe onde está a série de preços base e estará a média móvel. ex: Precos
#   'origin': nome da coluna onde está a série de preços base. ex: 'MGLU3.SA'

#GetDFPrices(AcoesBR, IndicesOuEUA, start, end, Ptype): retorna um dataframe com n séries de preços.
#   'AcoesBR': lista de tickers de ações brasileiras (que o Yahoo Finance precisa de '.SA' para interpretar), ex: ['JBSS3'] ou vazio []
#   'IndicesOuEUA': lista de tickers de ações ou ickers que não recebem '.SA'. ex: ['^BVSP','AAL']
#   'start': string da data de início da série. ex: 'DD/MM/AAAA' ou '5/12/2013'
#   'end': string da data final da série. ex: 'DD/MM/AAAA' ou '5/12/2013' ou '' para dia de hoje.
#   'Ptype': preço a ser retornado. 'Adj Close' OU 'Close'

#PVol(dfname, mode=): retorna a volatilidade anual de uma série de retornos
#   'dfname': nome do DataFrame
#   'mode': opcional, deixar vazio caso seja uma série de retornos. 'Price' caso seja uma série de preços

#PVar(dfname, mode=): retorna a variância de uma série de retornos.
#   'dfname': nome do dataframe
#   'mode': opcional, deixar vazio caso seja uma série de retornos. 'Price' caso seja uma série de preços

#PLogR(dfname): retorna os log retornos de um dataframe de preços
#   'dfname': nome do dataframe

#GetBacenData(Titulos, codigos_bcb): retorna um Precos com os dados retirados do API do BCB
#   'Titulos': lista de dados
#   'codigos_bcb': lista de códigos na mesma ordem que os Títulos
#
#   Códigos podem ser encontrados no site https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do?method=prepararTelaLocalizarSeries

#Objetos:
#series(ticker, Ptype, start, end): retorna a série de preços de uma ação
#   'ticker': ticker da ação ou índice no formato Yahoo. ex: 'JBSS3.SA', '^BVSP'
#   'Ptype': preço a ser retornado. 'Adj Close' OU 'Close'
#   'start': string da data de início da série. ex: 'DD/MM/AAAA' ou '5/12/2013'
#   'end': string da data final da série. ex: 'DD/MM/AAAA' ou '5/12/2013' ou '' para dia de hoje.
#   "self.prices" = série de preços
#   "self.ticker" = 'ticker'
#   "self.Ptype" = 'Ptype'
#   DropColumns(self): elimina todas as colunas que não são 'Data' ou Ptype. Muda "self.prices"
#   DropNaN(self): elimina valores em branco ou inválidos. Muda "self.prices"
#   Round(self): arredonda a coluna de preços. Muda "self.prices"
#   Treat(self): executa as 3 funções acima, muda o nome da série para "self.ticker" e retorna "self.prices"
#   ToCSV(self): salva a série de preços para um arquivo CSV chamado "self.ticker".csv

import pandas as pd
import pandas_datareader.data as web
import datetime as dt
from datetime import date
import numpy as np
import math

###########################################################################################################
def PresentValue(list, rate):
    a = 0
    x = 0
    while a < len(list):
        x = x + list[a]/pow(1 + rate, a)
        a += 1
    x = round(x, 2)
    return x

##########################################################################################################
##########################################################################################################
class series:
    def __init__(self, ticker, Ptype, start, end):
        self.ticker = ticker
        self.Ptype = Ptype
        self.start = start
        self.end = end

        if self.Ptype == 'Close':
            self.xyz = ['High', 'Low', 'Open', 'Adj Close', 'Volume']
        else:
            self.xyz = ['High', 'Low', 'Open', 'Close', 'Volume']

        self.prices = web.DataReader(ticker, 'yahoo', self.start, self.end)

    def DropColumns(self):
        self.prices.drop(self.xyz, axis=1, inplace=True)

    def DropNaN(self):
        self.prices.dropna(subset=[self.Ptype], axis=0, inplace=True)

    def Round(self):
        self.prices[self.Ptype] = self.prices[self.Ptype].round(2)

    def ToCSV(self):
        self.prices.to_csv(self.ticker + '.csv')

    def Treat(self):
        self.DropNaN()
        self.DropColumns()
        self.Round()
        self.prices.columns = [self.ticker]
        return self.prices

#################################################################################
def GetDFPrices(Tickers, start, end, Ptype):

    if isinstance(start, str):
        sday, smonth, syear = map(int, start.split('/'))
        start = dt.datetime(syear, smonth, sday)
    elif isinstance(start, dt.date):
        pass
    else:
        raise ValueError('As datas tem que estar no formato str dd/mm/yyy ou no formate dt.date')

    if isinstance(end, str):
        if end == '':
            end = date.today()
        else:
            eday, emonth, eyear = map(int, end.split('/'))
            end = dt.datetime(eyear, emonth, eday)
                         
    elif isinstance(end, dt.date):
            pass
    else:                 
        raise ValueError('As datas tem que estar no formato str dd/mm/yyy ou no formate dt.date')
                         
    df1 = series(Tickers[0], Ptype, start, end).Treat()

    i = 1

    while i < len(Tickers):
        s = series(Tickers[i], Ptype, start, end).Treat()
        df1 = df1.join(s)

        i += 1

    return df1

#####################################################################################################
#####################################################################################################
def Names(AtivosBR=None, Ativos=None):
    if AtivosBR is None:
        AtivosBR = []
    if Ativos is None:
        Ativos = []

    AtivosBR = [ativo + '.SA' for ativo in AtivosBR]
    return AtivosBR + Ativos

#####################################################################################################
#####################################################################################################
def SetAVKey(key):
    global AVkey
    AVkey = str(key)

##########################################################################################################
##########################################################################################################
class seriesAV:
    def __init__(self, ticker, Ptype, start, end):
        self.ticker = ticker
        
        if Ptype == 'Adj Close':
            self.Ptype = 'close'
        elif Ptype == 'Open':
            self.Pytpe = 'open'
        else:
            raise ValueError('O Ptype tem que ser Adj Close ou Open, se quiser adiconar mais um, trate de abrir o Impactus_Financial_Models e alterar')
            
        self.start = start
        self.end = end

        if self.Ptype == 'open':
            self.xyz = ['high', 'low', 'close', 'volume']
        elif self.Ptype == 'close':
            self.xyz = ['high', 'low', 'open', 'volume']
        
        self.prices = web.DataReader(ticker, 'av-daily', self.start, self.end, api_key=AVkey)

    def DropColumns(self):
        self.prices.drop(self.xyz, axis=1, inplace=True)

    def DropNaN(self):
        self.prices.dropna(subset=[self.Ptype], axis=0, inplace=True)

    def Round(self):
        self.prices[self.Ptype] = self.prices[self.Ptype].round(2)

    def ToCSV(self):
        self.prices.to_csv(self.ticker + '.csv')

    def Treat(self):
        self.DropNaN()
        self.DropColumns()
        self.Round()
        self.prices.columns = [self.ticker]
        return self.prices

#################################################################################
class seriesAVForex:
    def __init__(self, ticker, Ptype, start, end):
        self.ticker = ticker
        
        if Ptype == 'Adj Close':
            self.Ptype = 'close'
        elif Ptype == 'Open':
            self.Pytpe = 'open'
        else:
            raise ValueError('O Ptype tem que ser Adj Close ou Open, se quiser adiconar mais um, trate de abrir o Impactus_Financial_Models e alterar')
            
        self.start = start
        self.end = end

        if self.Ptype == 'open':
            self.xyz = ['high', 'low', 'close']
        elif self.Ptype == 'close':
            self.xyz = ['high', 'low', 'open']
        print(self.ticker)
        self.fr, self.to = map(str, self.ticker.split('/'))

        url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={self.fr}&to_symbol={self.to}&outputsize=full&apikey={AVkey}'
        
        ser = pd.read_json(url).iloc[6:, 1].iloc[::-1]
        self.prices = pd.DataFrame(ser.to_list(), index=ser.index)
        self.prices.columns = ['open', 'high', 'low', 'close']
        self.prices.index = pd.to_datetime(self.prices.index)
        self.prices = self.prices.loc[self.start:self.end, :]
        print(self.prices, '\n', self.Ptype)

    def DropColumns(self):
        self.prices.drop(self.xyz, axis=1, inplace=True)

    def DropNaN(self):
        self.prices.dropna(subset=[self.Ptype], axis=0, inplace=True)

    def Round(self):
        print(self.prices[self.Ptype], type(self.prices[self.Ptype]))
        self.prices[self.Ptype] = self.prices[self.Ptype].round(2)

    def ToCSV(self):
        self.prices.to_csv(self.ticker + '.csv')

    def Treat(self):
        self.DropNaN()
        self.DropColumns()
        self.prices.columns = [self.ticker]
        return self.prices

#################################################################################
def GetAVPrices(Tickers=None, Forex=None, start=None, end=None, Ptype=None):

    if isinstance(start, str):
        sday, smonth, syear = map(int, start.split('/'))
        start = dt.datetime(syear, smonth, sday)
    elif isinstance(start, dt.date):
        pass
    else:
        raise ValueError('As datas tem que estar no formato str dd/mm/yyy ou no formate dt.date')

    if isinstance(end, str):
        if end == '':
            end = date.today()
        else:
            eday, emonth, eyear = map(int, end.split('/'))
            end = dt.datetime(eyear, emonth, eday)
                         
    elif isinstance(end, dt.date):
            pass
    else:                 
        raise ValueError('As datas tem que estar no formato str dd/mm/yyy ou no formate dt.date')

    if Tickers is not None:
      df1 = seriesAV(Tickers[0], Ptype, start, end).Treat()
      i = 1
      
      while i < len(Tickers):
          s = seriesAV(Tickers[i], Ptype, start, end).Treat()
          df1 = df1.join(s)

          i += 1
      print(df1)
      if Forex is not None:
        j = 0
        while j < len(Forex):
            s = seriesAVForex(Forex[j], Ptype, start, end).Treat()
            df1 = df1.join(s)

            j += 1
    
    elif Forex is not None:
      df1 = seriesAVForex(Forex[0], Ptype, start, end).Treat()
      j = 1
      while j < len(Forex):
          s = seriesAVForex(Forex[j], Ptype, start, end).Treat()
          df1 = df1.join(s)

          j += 1

    else:
      raise ValueError('Tem que querer algum dos dois.')
    
    return df1

#####################################################################################################
#####################################################################################################
def AV_Names(AtivosBR=None, Ativos=None):
    if AtivosBR is None:
        AtivosBR = []
    if Ativos is None:
        Ativos = []

    AtivosBR = [ativo + '.SAO' for ativo in AtivosBR]
    return AtivosBR + Ativos

#####################################################################################################
#####################################################################################################
def MA(nPeriods, dfname, origin):
    dfname[str(nPeriods) + 'ma'] = dfname[str(origin)].rolling(window=nPeriods, min_periods=0).mean()
    dfname[str(nPeriods) + 'ma'] = dfname[str(nPeriods) + 'ma'].round(2)

#####################################################################################################
#####################################################################################################
def PVol(dfname, mode=None):
    if mode is None:
        return dfname.std()*np.sqrt(252)
    if mode == 'Price':
        return dfname.pct_change().std()*np.sqrt(252)

#####################################################################################################
#####################################################################################################
def VolEWMA(dfname, mode=None):
    if mode is None:
        return dfname.ewm(alpha=0.94).std().iloc[-1] * np.sqrt(252)

#####################################################################################################
#####################################################################################################
def PVar(dfname, mode=None):
    if mode is None:
        return dfname.var()
    if mode == 'Price':
        return dfname.pct_change().var()

#####################################################################################################
#####################################################################################################
def PLogR(dfname):
    return dfname.pct_change().apply(lambda x: np.log(1+x))

#####################################################################################################
#####################################################################################################
def GetFredData(ticker, start, end):
    sday, smonth, syear = map(int, start.split('/'))
    start = dt.datetime(syear, smonth, sday)

    if end == '':
        end = date.today()

    else:
        eday, emonth, eyear = map(int, end.split('/'))
        end = dt.datetime(eyear, emonth, eday)

    return web.DataReader(ticker, 'fred', start, end)

#####################################################################################################
#####################################################################################################
def GetEcondbData(query, start, end, coluna):
    sday, smonth, syear = map(int, start.split('/'))
    start = dt.datetime(syear, smonth, sday)

    if end == '':
        end = date.today()

    else:
        eday, emonth, eyear = map(int, end.split('/'))
        end = dt.datetime(eyear, emonth, eday)

    df = web.DataReader(query, 'econdb', start, end).iloc[:, coluna].to_frame()

    return df

#####################################################################################################
#####################################################################################################
def GetBacenData(Titulos, codigos_bcb, Start, End):
    main_df = pd.DataFrame()

    for i, codigo in enumerate(codigos_bcb):
        url = f'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{str(codigo)}/dados?formato=json&dataInicial={Start}&dataFinal={End}'
        df = pd.read_json(url)

        df['DATE'] = pd.to_datetime(df['data'], dayfirst=True)
        df.drop('data', axis=1, inplace=True)
        df.set_index('DATE', inplace=True)
        df.columns = [str(Titulos[i])]

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.merge(df, how='outer', left_index=True, right_index=True)

    main_df.fillna(method='ffill', inplace=True)

    return main_df

#####################################################################################################
#####################################################################################################
def VaR(serie, confianca, mode=None):
    if mode is None:
        return np.exp(serie.quantile(1 - (confianca/100))) - 1
    
    elif mode == 'Price':
        return serie.pct_change().quantile(1 - (confianca/100))
    
    elif mode == 'LinRet':
        return serie.quantile(1 - (confianca/100))
    
    else:
      raise ValueError('mode tem que ser Price ou LinRet!')
#####################################################################################################
#####################################################################################################
def CVaR(serie, confianca, mode=None):
    if mode is None:
        serie = serie.apply(lambda x: np.exp(x) - 1)
    
    elif mode == 'Price':
        serie = serie.pct_change()
    
    elif mode == 'LinRet':
        pass
    
    else:
      raise ValueError('mode tem que ser Price ou LinRet!')

    x = (100 - confianca) / 100

    serie = serie.sort_values()
    y = round(len(serie) * x)

    return serie.iloc[0:y].mean()

#####################################################################################################
#####################################################################################################
def Sharpe(dfname, mode=None):

    if mode is None:
        df = dfname

    elif mode == 'Price':
        df = PLogR(dfname)

    elif mode == 'LinRet':
        df = dfname.apply(lambda x: np.log(x + 1))

    Sharpe = df.sum(axis=0) / (df.std(axis=0) * np.sqrt(252))

    return Sharpe

#####################################################################################################
#####################################################################################################
def UnderWater(dfname):
    uw = pd.DataFrame(index=dfname.index, columns=dfname.columns)

    for coluna in dfname.columns:
        for i in range(0, len(dfname)):
            uw[coluna][i] = (dfname[coluna].iloc[i] - dfname[coluna].iloc[0:i + 1].max()) / dfname[coluna].iloc[0:i + 1].max()

    return uw

#####################################################################################################
#####################################################################################################
def Beta(Retornos, BenchMark, mode=None):
    if mode is None:
      rets = Retornos
      BM = BenchMark
    elif mode == 'Price':
      rets = PLogR(dfname)
      BM = PLogR(BenchMark)

    Beta = rets.cov(BM) / BM.var()

    return Beta
#####################################################################################################
#####################################################################################################
def RollBeta(dfname, Indice, mode=None, window=100):
    if mode is None:
        df = dfname
        Ind = Indice
    elif mode == 'Price':
        df = PLogR(dfname)
        Ind = PLogR(Indice)

    RB = pd.DataFrame()
    for coluna in df.columns:
        RB[coluna] = df[coluna].rolling(window=window, min_periods=window).cov(Ind.iloc[:, 0]) / Ind.iloc[:, 0].rolling(window=window, min_periods=window).var()

    RB.index = df.index

    return RB

#####################################################################################################
#####################################################################################################
def RollVol(dfname, mode=None, window=100):
    if mode is None:
        df = dfname
    elif mode == 'Price':
        df = PLogR(dfname)

    Vol = pd.DataFrame()
    for coluna in df.columns:
        Vol[coluna] = df[coluna].rolling(window=window, min_periods=0).std()
    Vol = Vol.apply(lambda x: x * np.sqrt(252))

    Vol.index = df.index

    return Vol

#####################################################################################################
#####################################################################################################
def GetDFPrices2(AcoesBR, IndicesOuEUA, start, end):
    tickers = []

    sday, smonth, syear = map(int, start.split('/'))
    start = dt.datetime(syear, smonth, sday)

    if end == '':
        end = date.today()

    else:
        eday, emonth, eyear = map(int, end.split('/'))
        end = dt.datetime(eyear, emonth, eday)

    for i, Acao in enumerate(AcoesBR):
        AcoesBR[i] = Acao.upper()
    AcoesBR.sort()
    for i, Acao in enumerate(IndicesOuEUA):
        IndicesOuEUA[i] = Acao.upper()
    IndicesOuEUA.sort()

    if len(AcoesBR) != 0:
        for item in AcoesBR:
            tickers.append(item + '.SA')

    if len(IndicesOuEUA) != 0:
        for item in IndicesOuEUA:
            tickers.append(item)

    dict = {}
    for ticker in tickers:
        dict[ticker] = web.DataReader(ticker, 'yahoo', start, end)

    main_df = pd.DataFrame()

    for ticker in tickers:
        df = dict[ticker]
        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

    return main_df

