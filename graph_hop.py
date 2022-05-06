import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import medfilt
from sklearn.linear_model import LinearRegression
# from scipy.odr import *
from scipy.optimize import leastsq

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


    def _doublegaussian(self, params, x):
        # params is a vector of the parameters:
        # params = [f_U, sigma_U, w_U, f_F, sigma_F, w_F]
        (c1, mu1, sigma1, c2, mu2, sigma2) = params
        res =   c1 * np.exp( - (x - mu1)**2.0 / (4.0 * sigma1**2.0) ) \
          + c2 * np.exp( - (x - mu2)**2.0 / (4.0 * sigma2**2.0) )
        return res
        # return B[2]/np.sqrt(2*np.pi*B[1])*np.exp(((x-B[0])/(2*B[1]))**2) + B[5]/np.sqrt(2*np.pi*B[4])*np.exp(((x-B[3])/(2*B[4]))**2)

    def _double_gaussian_fit(self, params):
        fitting = self._doublegaussian(params, self.bin)
        return (fitting - self.values_histogram_bins_proc)

    def fit(self, guess:list):
        fitting = leastsq(self._double_gaussian_fit, guess)
        self._fit_plot(fitting)
        return fitting
    
    

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
        plt.xlabel('$f_y$(pN)')
        plt.ylabel('$p(f)\:[1/pN]$')
        plt.title(self.name+ ': Force Histogram')
        # self.data_frame['Y_force'].hist(grid=False, bins=rice)
        self.values_histogram_bins, bins, patches = plt.hist(self.data_frame['Y_force'], density=True, bins=rice, orientation='horizontal', label='Force Y', stacked=True) # y
        self.bin = [(bins[i+1] + bins[i])/2 for i in range(len(bins)-1)] # x
        self.values_histogram_bins_proc = np.copy(self.values_histogram_bins)
        self.values_histogram_bins_proc[self.values_histogram_bins_proc < 0.05] = 0.0
        plt.legend()
        plt.show()
    
    def _fit_plot(self, fitting):
        rice = int(6*np.cbrt(self.data_frame.shape[0]))
        plt.hist(self.data_frame['Y_force'], density=True, bins=rice, orientation='horizontal', label='Force Y', stacked=True)
        plt.plot(self._doublegaussian(fitting[0], self.bin), self.bin, c='r', label='Fit')
        plt.axhline(y = fitting[0][1], color = 'g', linestyle = 'dashed', label = '$\mu_1$')
        plt.axhline(y = fitting[0][4], color = 'y', linestyle = 'dashed', label = '$\mu_2$')    
        plt.ylabel('$f_y$(pN)')
        plt.xlabel('$p(f)\:[1/pN]$')
        plt.title(self.name+ ': Force Histogram + Fit')
        plt.legend()
        plt.show()
        print(f"c_1 = {fitting[0][0]}, mu_1 = {fitting[0][1]}, sigma_1 = {fitting[0][2]}")
        print(f"c_2 = {fitting[0][3]}, mu_2 = {fitting[0][4]}, sigma_2 = {fitting[0][5]}")


    def _prova(self):
        plt.plot(self.bin, self.values_histogram_bins)
        plt.show()

    def subplots(self, fitting):
        df_mean = self.data_frame['Y_force'].mean()
        df_std = self.data_frame['Y_force'].std()
        # Setting up the plot surface
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(nrows=1, ncols=3)
        # First axes
        ax0 = fig.add_subplot(gs[0, :2])
        ax0.plot(self.data_frame['time(sec)'], self.data_frame['Y_force'], label='Y_force')
        ax0.axhline(y = df_mean, color = 'b', linestyle = 'dashed', label = '$\mu$')    
        ax0.axhline(y = df_mean+3*df_std, color = 'r', linestyle = 'dashed', label = '$\mu\pm3\sigma$')   
        ax0.axhline(y = df_mean-3*df_std, color = 'r', linestyle = 'dashed')
        ax0.set_ylabel('$f_y$(pN)')
        ax0.set_xlabel('$t$(s)')
        ax0.set_title(self.name)
        ax0.legend()
        # Second axes
        rice = int(6*np.cbrt(self.data_frame.shape[0]))
        ax1 = fig.add_subplot(gs[0, 2])
        rice = int(6*np.cbrt(self.data_frame.shape[0]))
        ax1.hist(self.data_frame['Y_force'], density=True, bins=rice, orientation='horizontal', label='Force Y', stacked=True)
        ax1.plot(self._doublegaussian(fitting[0], self.bin), self.bin, c='r', label='Fit')
        ax1.axhline(y = fitting[0][1], color = 'g', linestyle = 'dashed', label = '$\mu_1$')
        ax1.axhline(y = fitting[0][4], color = 'y', linestyle = 'dashed', label = '$\mu_2$')    
        # ax1.set_ylabel('$f_y$(pN)')
        ax1.set_xlabel('$p(f)\:[1/pN]$')
        ax1.set_title(self.name+ ': Force Histogram + Fit')
        ax1.legend()
        
        plt.show()