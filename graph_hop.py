import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from sklearn.linear_model import LinearRegression
from scipy.odr import *

class Graph_hop:
    def __init__(self, dir_name, number, number_file) -> None:
        self.dir_name = dir_name
        self.number = str(number)
        self.number_file = str(number_file)
        self.name = self.dir_name + '/' + self.number + '_' + self.number_file 
        
    def do_graph(self, time_range=None):
        data = []
        data.append(pd.read_fwf(self.name+'.txt', colspecs = [(0, 9), (9, 17), (17, 28), (28, 37), (37, -1)]))
        self.data_frame = pd.concat([j for j in data], ignore_index=True) # I add together all the datasets
        '''
            I would like to ask the teacher how to implement this thing
        '''
        # self._rebin()
        self.data_frame['time(sec)'] = self.data_frame['time(sec)'].sub(self.data_frame['time(sec)'].loc[0]) 
        # print(self.data_frame['time(sec)'])
        # self.data_frame.plot.line(x ='lambda', y='Y_force')
        if time_range:
            time_range[0] += 1.95e-3
            time_range[1] -= 1.95e-3
            lista = []
            lista.append(self.data_frame.index[self.data_frame['time(sec)'] == time_range[0]].tolist()[0])
            lista.append(self.data_frame.index[self.data_frame['time(sec)'] == time_range[1]].tolist()[0])
            self.data_frame = self.data_frame.loc[lista[0]:lista[1]-1]
            self.data_frame = self.data_frame.reset_index()
            self.data_frame = self.data_frame.drop(['index'], axis=1)
       
        self.graph()
        self.histogram()

        return self.data_frame


    def _doublegaussian(self, B, x):
        # B is a vector of the parameters:
        # B = [f_U, sigma_U, w_U, f_F, sigma_F, w_F]
        return B[2]/np.sqrt(2*np.pi*B[1])*np.exp(((x-B[0])/(4*B[1]))**2) + B[5]/np.sqrt(2*np.pi*B[4])*np.exp(((x-B[3])/(4*B[4]))**2)

    def fit(self, lista_fit_gaussiano, y, guess:list):
        # double gaussian fit
        self.x = np.linspace(lista_fit_gaussiano[0], lista_fit_gaussiano[1], num=y.shape[0])
        mydata = RealData(self.x, y)
        doubgauss = Model(self._doublegaussian)
        myodr = ODR(mydata, doubgauss, beta0=guess)
        self.myoutput = myodr.run()
        self.myoutput.pprint()
        # covariana_matrice = myoutput.cov_beta
        self.sigma_diagonale = np.sqrt(np.diag(self.myoutput.cov_beta))
        self._fit_plot()
        return self.myoutput.beta, self.sigma_diagonale

    def graph(self):
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

    def histogram(self):
        rice = int(6*np.cbrt(self.data_frame.shape[0]))
        # scott = int(3.49*self.data_frame['Y_force'].std()/np.cbrt(self.data_frame.shape[0]))
        plt.ylabel('$f_y$(pN)')
        plt.xlabel('bins')
        plt.title(self.name+ ' Force Histogram')
        # self.data_frame['Y_force'].hist(grid=False, bins=rice)
        plt.hist(self.data_frame['Y_force'], density=True, bins=rice, orientation='horizontal', label='Force Y')
        # self._fit_plot()
        plt.legend()
    
    def _fit_plot(self):
        gaussianadoppia = np.vectorize(self._doublegaussian)
        y_pred = gaussianadoppia(self.myoutput, self.x)
        plt.plot(self.x, y_pred, color='r', label='Fit')
