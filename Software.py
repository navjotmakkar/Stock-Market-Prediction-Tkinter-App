import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import tkinter as tk
from tkinter import ttk
from tkinter import IntVar
import pandas as pd
import numpy as np
import pylab
import sys
import csv
import re
import requests
import codecs
import datetime
from time import sleep
import math
#from sklearn import preprocessing, cross_validation
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import threading
import arrow

matplotlib.use('TkAgg')
matplotlib.rcParams.update({'font.size':9})
style.use('ggplot')

Large_Font = ('Verdana', 12)
Medium_Font = ('Verdana', 10)
Small_Font = ('Verdana', 8)

ticker = 'AAPL'
exchange = 'NASDAQ'

fig = plt.figure(facecolor='#07000d')

def get_quote_data(symbol, data_range, data_interval):
    res = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'.format(**locals()))
    data = res.json()
    body = data['chart']['result'][0]    
    dt = datetime.datetime
    dt = pd.Series(map(lambda x: arrow.get(x).to('Asia/Calcutta').datetime.replace(tzinfo=None), body['timestamp']), name='Date')
    df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
    colT=['open', 'high', 'low', 'close', 'volume']
    df.reindex(columns=colT)
    df.rename(columns={'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
    df.dropna(inplace=True) #removing NaN rows
    
    return df.loc[:, ('Open', 'High', 'Low', 'Close', 'Volume')]

def get_data_from_google(ticker, exchange, period=60, days=1):
    
    uri = 'http://www.google.com/finance/getprices?x={exchange}&i={period}&p={days}d&f=d,o,h,l,c,v&df=cpct&q={ticker}'.format(ticker=ticker, period=period, days=days, exchange=exchange)
    page = requests.get(uri)
    reader = csv.reader(codecs.iterdecode(page.content.splitlines(), "utf-8"))
    columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    rows = []
    times = []
    
    
    for row in reader:
        if re.match('^[a\d]', row[0]):
            if row[0].startswith('a'):
                start = datetime.datetime.fromtimestamp(int(row[0][1:]))
                times.append(start)
            else:
                times.append(start+datetime.timedelta(seconds=period*int(row[0])))
            rows.append(map(float, row[1:]))
    
    if len(rows):
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'), columns=columns)
    else:
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'))

def get_data_from_google_daily(ticker, exchange, period=86400, years=1):
    
    uri = 'http://www.google.com/finance/getprices?x={exchange}&i={period}&p={years}Y&f=d,o,h,l,c,v&df=cpct&q={ticker}'.format(ticker=ticker, period=period, years=years, exchange=exchange)
    page = requests.get(uri)
    reader = csv.reader(codecs.iterdecode(page.content.splitlines(), "utf-8"))
    columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    rows = []
    times = []
    
    for row in reader:
        if re.match('^[a\d]', row[0]):
            if row[0].startswith('a'):
                start = datetime.datetime.fromtimestamp(int(row[0][1:]))
                times.append(start)
            else:
                times.append(start+datetime.timedelta(seconds=period*int(row[0])))
            rows.append(map(float, row[1:]))
    
    if len(rows):
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'), columns=columns)
    else:
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'))

def RSI(dataframe, period):
    delta = dataframe.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) / \
         pd.stats.moments.ewma(d, com=period-1, adjust=False)
    return 100 - 100 / (1 + rs)

def dataset_prep(df):
    
    df_cs = df.copy(deep=True)
    cols_to_drop = ['Volume']
    df_cs.drop(cols_to_drop, axis=1, inplace=True)
    df_cs.reset_index(inplace=True)
    df_cs['Date'] = df_cs['Date'].map(mdates.date2num)
    
    volumeMin = df['Volume'].min()
    
    sma1 = pd.stats.moments.rolling_mean(df['Close'], window=12)
    sma2 = pd.stats.moments.rolling_mean(df['Close'], window=26)
    df['SMA1'] = sma1
    df['SMA2'] = sma2
    
    df['RSI']= RSI(df['Close'], 14)
    
    df['EWMA12'] = pd.ewma(df['Close'], span=12)
    df['EWMA26'] = pd.ewma(df['Close'], span=26)
    df['MACD'] = df['EWMA12'] - df['EWMA26']
    df['MACD_EWMA9'] = pd.ewma(df['MACD'], span=9)
    
    return df, df_cs

def graphData():
    
    fig.clf()
    
    #df_temp = get_data_from_google(ticker, exchange)
    
    df_temp = get_quote_data(ticker, '1y', '1d')
    df, df_cs = dataset_prep(df_temp)
    
    ax0 = plt.subplot2grid((7,4), (0,0), rowspan=1, colspan=4, axisbg='#07000d')
    
    ax0.plot(df_cs['Date'], df['RSI'], '#1a8782', linewidth=1.2)
    ax0.fill_between(df_cs['Date'], df['RSI'], 70, where=(df['RSI']>=70), facecolor='#8f2020', edgecolor='#8f2020')
    ax0.fill_between(df_cs['Date'], df['RSI'], 30, where=(df['RSI']<=30), facecolor='#386d13', edgecolor='#386d13')
    
    ax0.xaxis_date()
    ax0.set_ylim(0, 100)
    ax0.axhline(70, color='#8f2020', linewidth=1)
    ax0.axhline(30, color='#386d13', linewidth=1)
    ax0.spines['bottom'].set_color('#5998ff')
    ax0.spines['top'].set_color('#5998ff')
    ax0.spines['left'].set_color('#5998ff')
    ax0.spines['right'].set_color('#5998ff')
    #ax0.text(0.015, 0.95, 'RSI (14)', va='top', color='w', transform=ax0.transAxes)
    ax0.tick_params(axis='x', colors='w')
    ax0.tick_params(axis='y', colors='w')
    ax0.set_yticks([30, 70])
    plt.ylabel('RSI', color='w')
    
    ax1 = plt.subplot2grid((7,4), (1,0), rowspan=4, colspan=4, sharex=ax0, axisbg='#07000d')
    
    sma1Label = '12 SMA'
    sma2Label = '26 SMA'
    
    candlestick_ohlc(ax1, df_cs.values, width=0.0001, colorup='#ff1717', colordown='#53c156')
    
    ax1.plot(df_cs['Date'], df['SMA1'], '#5998ff', label=sma1Label, linewidth=1.5)
    ax1.plot(df_cs['Date'], df['SMA2'], '#e1edf9', label=sma2Label, linewidth=1.5)
    
    ax1.xaxis_date()
    ax1.grid(True, color='w')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.yaxis.label.set_color('w')
    ax1.spines['bottom'].set_color('#5998ff')
    ax1.spines['top'].set_color('#5998ff')
    ax1.spines['left'].set_color('#5998ff')
    ax1.spines['right'].set_color('#5998ff')
    ax1.tick_params(axis='y', colors='w')
    plt.ylabel('Stock Price')
    
    Leg = plt.legend(loc=9, ncol=2, prop={'size':7}, fancybox=True, borderaxespad=0)
    Leg.get_frame().set_alpha(0.4)
    LegText = pylab.gca().get_legend().get_texts()
    pylab.setp(LegText[0:], color='w')
    
    
    ax2 = plt.subplot2grid((7,4), (5,0), rowspan=1, colspan=4, sharex=ax0, axisbg='#07000d')
    
    volumeMin = df['Volume'].min()
    
    ax2.plot(df_cs['Date'], df['Volume'], '#00ffe8', linewidth=0.8)
    ax2.fill_between(df_cs['Date'], volumeMin, df['Volume'], facecolor='#00ffe8', alpha=0.5)
    
    ax2.xaxis_date()
    ax2.axes.yaxis.set_ticklabels([])
    ax2.grid(False)
    ax2.spines['bottom'].set_color('#5998ff')
    ax2.spines['top'].set_color('#5998ff')
    ax2.spines['left'].set_color('#5998ff')
    ax2.spines['right'].set_color('#5998ff')
    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    plt.ylabel('Volume', color='w')
    
    
    ax3 = plt.subplot2grid((7,4), (6,0), rowspan=1, colspan=4, sharex=ax0, axisbg='#07000d')
    
    ax3.plot(df_cs['Date'], df['MACD'], color='#4ee6fd', linewidth=1.5)
    ax3.plot(df_cs['Date'], df['MACD_EWMA9'], color='#e1edf9', linewidth=1)
    ax3.fill_between(df_cs['Date'], df['MACD']-df['MACD_EWMA9'], 0, alpha=0.5, facecolor='#00ffe8', edgecolor='#00ffe8')
    
    ax3.xaxis_date()
    ax3.spines['bottom'].set_color('#5998ff')
    ax3.spines['top'].set_color('#5998ff')
    ax3.spines['left'].set_color('#5998ff')
    ax3.spines['right'].set_color('#5998ff')
    ax3.tick_params(axis='x', colors='w')
    ax3.tick_params(axis='y', colors='w')
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax3.xaxis.get_ticklabels():
        label.set_rotation(45)
    plt.ylabel('MACD', color='w')
    
    plt.xlabel('Date', color='w')
    plt.suptitle(ticker, color='w')
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.subplots_adjust(left=0.09, bottom=0.14, right=0.95, top=0.94, wspace=0.20, hspace=0)

def animate(i):
    
    graphData()

def SVM_pred():
    
    df = get_data_from_google_daily(ticker, exchange, period=86400, years=40)
    data = get_quote_data(ticker, '15y', '1d')
    
    df.dropna(inplace=True)
    
    df_temp = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    df_temp['HLP'] = (df_temp['High'] - df_temp['Close']) / df_temp['Close'] * 100.0
    df_temp['Change'] = (df_temp['Close'] - df_temp['Open']) / df_temp['Open'] * 100
    df_temp['EWMA12']= pd.ewma(df_temp['Close'], span=12)
    df_temp['EWMA26']= pd.ewma(df_temp['Close'], span=26)
    df_temp['MACD']= (df_temp['EWMA12'] - df_temp['EWMA26'])
    df_temp['RSI']= RSI(df_temp['Close'],6)
    
    df_temp = df_temp[['Close', 'HLP', 'Change', 'Volume', 'EWMA12', 'EWMA26', 'MACD', 'RSI']]

    forecast_col = 'Close'
    forecast_out = 8
    df_temp['Label'] = df_temp[forecast_col].shift(-forecast_out)
    df_pred = df_temp[-8:]
    df_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_temp.dropna(inplace=True)
    X = df_temp.drop(['Label'], 1)
    Y = df_temp['Label']
    
    y_norm = preprocessing.scale(Y)
    x_norm = preprocessing.scale(X.values)
    
    rows=x_norm.shape[0]
    a=int(rows*0.80)
    
    X_train=x_norm[:a]
    Y_train=y_norm[:a]
    X_test=x_norm[a:]
    Y_test=y_norm[a:]
    
    y_orig = data['Close'][-8:].iloc[:].values
    y_orig_mean = y_orig.mean(axis = 0)
    y_orig_std = y_orig.std(axis = 0)
    
    clf1 = svm.SVR(kernel = 'rbf', gamma = 0.0001, C = 1000, epsilon = 0.001)
    clf1.fit(X_train, Y_train)
    accuracy = clf1.score(X_test, Y_test)
    
    x_pred = preprocessing.scale(df_pred.drop(['Label'], 1))
    y_pred_clf1  = clf1.predict(x_pred)
    y_new_clf1 = (y_pred_clf1*y_orig_std) + y_orig_mean
    
    dates = []
    for i in range(0, 8):
        date = data.index[-1] + datetime.timedelta(days=i)
        dates.append(date)
    
    return accuracy, dates, y_new_clf1

def NN_pred():
    
    df = get_data_from_google_daily(ticker, exchange, period=86400, years=40)
    data = get_quote_data(ticker, '15y', '1d')
    df.dropna(inplace=True)
    
    df_temp = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    df_temp['HLP'] = (df_temp['High'] - df_temp['Close']) / df_temp['Close'] * 100.0
    df_temp['Change'] = (df_temp['Close'] - df_temp['Open']) / df_temp['Open'] * 100
    df_temp['EWMA12']= pd.ewma(df_temp['Close'], span=12)
    df_temp['EWMA26']= pd.ewma(df_temp['Close'], span=26)
    df_temp['MACD']= (df_temp['EWMA12'] - df_temp['EWMA26'])
    df_temp['RSI']= RSI(df_temp['Close'],6)
    
    df_temp = df_temp[['Close', 'HLP', 'Change', 'Volume', 'EWMA12', 'EWMA26', 'MACD', 'RSI']]

    forecast_col = 'Close'
    forecast_out = 8
    df_temp['Label'] = df_temp[forecast_col].shift(-forecast_out)
    df_pred = df_temp[-8:]
    df_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_temp.dropna(inplace=True)
    X = df_temp.drop(['Label'], 1)
    Y = df_temp['Label']
    
    
    y_norm = preprocessing.scale(Y)
    x_norm = preprocessing.scale(X.values)
    
    rows=x_norm.shape[0]
    a=int(rows*0.80)
    
    X_train=x_norm[:a]
    Y_train=y_norm[:a]
    X_test=x_norm[a:]
    Y_test=y_norm[a:]
    
    y_orig = data['Close'][-8:].iloc[:].values
    y_orig_mean = y_orig.mean(axis = 0)
    y_orig_std = y_orig.std(axis = 0)
    
    neu = MLPRegressor(hidden_layer_sizes = 1000, activation = 'tanh', solver = 'lbfgs' , alpha=0.001, batch_size = 100, learning_rate = 'adaptive')
    neu.fit(X_train, Y_train)
    accuracy = neu.score(X_test, Y_test)
    
    x_pred = preprocessing.scale(df_pred.drop(['Label'], 1))
    y_pred_neu  = neu.predict(x_pred)
    y_new_neu = (y_pred_neu*y_orig_std) + y_orig_mean
    
    dates = []

    for i in range(0, 8):

        date = data.index[-1] + datetime.timedelta(days=i)
        dates.append(date)
    
    return accuracy, dates, y_new_neu

def popup(page, var):
    
    app1 = page()
    app1.mainloop()

class Software(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Stock Analyser")
        
        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        menubar = tk.Menu(container)
        
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Change Stock', command=lambda: self.show_frame(Selection))
        filemenu.add_separator()
        filemenu.add_command(label='Exit', command=quit)
        menubar.add_cascade(label='File', menu=filemenu)
        
        NN_var = IntVar()
        SVM_var = IntVar()
        
        predmenu = tk.Menu(menubar, tearoff=1)
        predmenu.add_checkbutton(label='Neural Network', variable=NN_var, onvalue=1, offvalue=0, command=lambda: popup(NeuralNet, NN_var))
        predmenu.add_checkbutton(label='SVM', variable=SVM_var, onvalue=1, offvalue=0, command=lambda: popup(SVM, SVM_var))
        menubar.add_cascade(label='Prediction', menu=predmenu)
        
        tk.Tk.config(self, menu=menubar)
        
        global menubar_off
        def menubar_off():
            menubar.entryconfig('File', state='disabled')
            menubar.entryconfig('Prediction', state='disabled')
        
        global filemenu_on
        def filemenu_on():
            menubar.entryconfig('File', state='normal')
        
        global predmenu_on
        def predmenu_on():
            menubar.entryconfig('Prediction', state='normal')
        
        self.frames = {}
        
        for F in (Disclaimer, Selection, Waiting, GraphPage, NeuralNetFrame):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')
        
        self.show_frame(Disclaimer)
    
    def show_frame(self, cont):
        
        frame = self.frames[cont]
        frame.tkraise()

class Disclaimer(tk.Frame):
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        
        menubar_off()
        
        def agree():
            controller.show_frame(Selection)
            filemenu_on()
        
        label1 = tk.Label(self, text='Stock Analyser', font=Large_Font)
        label1.pack(pady=10, padx=10)
        
        label2 = tk.Label(self, text='\nDisclaimer', font=Medium_Font)
        label2.pack(pady=10, padx=10)
        
        label3 = tk.Label(self, text='\nUse the software at your own risk.', font=Small_Font)
        label3.pack(pady=10, padx=10)
        
        button1 = ttk.Button(self, text='Agree', command=agree)
        button1.pack()
        
        button2 = ttk.Button(self, text="Disagree", command=quit)
        button2.pack()

class Selection(tk.Frame):
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        
        label1 = tk.Label(self, text='Stock Name', font=Medium_Font)
        label1.pack(pady=10, padx=10)
        
        entry1 = ttk.Entry(self)
        entry1.pack()
        entry1.focus_set()
        
        label2 = tk.Label(self, text='Exchange', font=Medium_Font)
        label2.pack(pady=10, padx=10)
        
        entry2 = ttk.Entry(self)
        entry2.pack()
        entry2.focus_set()
        
        def callback():
            
            global ticker
            global exchange
            
            ticker = (entry1.get())
            exchange = (entry2.get())
            
            controller.show_frame(Waiting)
        
        button1 = ttk.Button(self, text='Submit', command=callback)
        button1.pack(pady=10)

class Waiting(tk.Frame):
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        
        def cont():
            controller.show_frame(GraphPage)
            predmenu_on()
        
        button1 = ttk.Button(self, text='Continue', command=cont)
        
        pb_value = 0
        progressbar = ttk.Progressbar(self, orient='horizontal', length=200, mode='determinate', variable=pb_value)
        progressbar.pack(side='top', pady=20)
        progressbar.start()
        def stop_progressbar():
            progressbar.stop()
            button1.pack(pady=10)
        self.after(20000, stop_progressbar)

class GraphPage(tk.Frame):
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        
        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class NeuralNet(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Neural Net Prediction")
        
        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        
        frame = NeuralNetFrame(container, self)
        self.frames[NeuralNetFrame] = frame
        frame.grid(row=0, column=0, sticky='nsew')
        
        self.show_frame(NeuralNetFrame)
    
    def show_frame(self, cont):
        
        frame = self.frames[cont]
        frame.tkraise()

class NeuralNetFrame(tk.Frame):
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        
        accuracy, dates, prediction = NN_pred()
        
        print(accuracy)
        
        tv = ttk.Treeview(self, height=8)
        tv['columns'] = ['date', 'predicted']
        tv.heading('date', text='Date')
        tv.column('date', width=150, anchor='center')
        tv.heading('predicted', text='Predicted Price')
        tv.column('predicted', width=150, anchor='center')
        
        for i in range(0, 8):
            tv.insert('', 'end', text=str(i+1)+'.', values=(dates[i], prediction[i]))
        
        tv.pack()

class SVM(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "SVM Prediction")
        
        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        
        frame = SVMFrame(container, self)
        self.frames[SVMFrame] = frame
        frame.grid(row=0, column=0, sticky='nsew')
        
        self.show_frame(SVMFrame)
    
    def show_frame(self, cont):
        
        frame = self.frames[cont]
        frame.tkraise()

class SVMFrame(tk.Frame):
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        
        accuracy, dates, prediction = SVM_pred()
        
        print(accuracy)
        
        tv = ttk.Treeview(self, height=8)
        tv['columns'] = ['date', 'predicted']
        tv.heading('date', text='Date')
        tv.column('date', width=150, anchor='center')
        tv.heading('predicted', text='Predicted Price')
        tv.column('predicted', width=150, anchor='center')
        
        for i in range(0, 8):
            tv.insert('', 'end', text=str(i+1)+'.', values=(dates[i], prediction[i]))
        
        tv.pack()

app = Software()
app.geometry("1280x720")
animate_object = animation.FuncAnimation(fig, animate, interval=5000)
app.mainloop()









