import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from sklearn.linear_model import LinearRegression
from scipy.odr import *
import math

class Graph_hop:
    def __init__(self, dir_name, number, number_file) -> None:
        self.dir_name = dir_name
        self.number = str(number)
        self.number_file = str(number_file)
        self.name = self.dir_name + '/' + self.number + '_' + self.number_file 
        
    def do_graph(self, time_range):
        data = []
        data.append(pd.read_fwf(self.name+'.txt', colspecs = [(0, 9), (9, 17), (17, 28), (28, 37), (37, -1)]))
        self.data_frame = pd.concat([j for j in data], ignore_index=True) # I add together all the datasets
        '''
            I would like to ask the teacher how to implement this thing
        '''
        # self._rebin()

        # self.data_frame.plot.line(x ='lambda', y='Y_force')
        if time_range:
            lista = []
            lista.append(self.data_frame.index[self.data_frame['time(sec)'] == time_range[0]].tolist()[0])
            lista.append(self.data_frame.index[self.data_frame['time(sec)'] == time_range[1]].tolist()[0])
            self.data_frame = self.data_frame.loc[lista[0]:lista[1]-1]
            self.data_frame = self.data_frame.reset_index()
            self.data_frame = self.data_frame.drop(['index'], axis=1)
       
        df_mean = self.data_frame['Y_force'].mean()
        df_std = self.data_frame['Y_force'].std()
        plt.plot(self.data_frame['time(sec)'], self.data_frame['Y_force'], label='Y_force')
        plt.axhline(y = df_mean, color = 'b', linestyle = 'dashed', label = '$\mu$')    
        plt.axhline(y = df_mean+3*df_std, color = 'r', linestyle = 'dashed', label = '$\mu\pm3\sigma$')   
        plt.axhline(y = df_mean-3*df_std, color = 'r', linestyle = 'dashed')   
        plt.ylabel('$f_y$(pN)')
        plt.xlabel('$t$(s)')
        plt.title(self.name)
        plt.legend()
        plt.show()

        return self.data_frame