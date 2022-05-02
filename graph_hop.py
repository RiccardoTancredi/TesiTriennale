import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from sklearn.linear_model import LinearRegression
from scipy.odr import *
import math

class Graph_hop:
    def __init__(self, dir_name, number) -> None:
        self.dir_name = dir_name
        self.number = str(number)
        self.name = self.dir_name + '/' + self.number + '.txt'
        
    def do_graph(self):
        data = []
        data.append(pd.read_fwf(self.name, colspecs = [(0, 9), (9, 17), (17, 28), (28, 37), (37, -1)]))
        self.data_frame = pd.concat([j for j in data], ignore_index=True) # I add together all the datasets
        '''
            I would like to ask the teacher how to implement this thing
        '''
        # self._rebin()
        
        # print(self.data_frame[['lambda', 'Y_force']])
        # self.data_frame.plot.line(x ='lambda', y='Y_force')
        plt.scatter(self.data_frame['time(sec)'], self.data_frame['Y_force'], label='Y_force')
        plt.ylabel('$f_y$(pN)')
        plt.xlabel('$t$(s)')
        plt.title(self.name)
        # f_F, f_U = self.maximum_f(False)
        # if f_F and f_U:
        #     plt.annotate(f'f_F', xy=(self.data_frame['lambda'][np.where(self.data_frame['Y_force'].values == f_F)[0][0]], f_F),) 
        #     plt.annotate(f'f_U', xy=(self.data_frame['lambda'][np.where(self.data_frame['Y_force'].values == f_U)[0][0]], f_U),) 
        plt.legend()
        plt.show()
        # return [f_F, f_U]