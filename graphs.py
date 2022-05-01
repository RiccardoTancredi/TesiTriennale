import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from sklearn.linear_model import LinearRegression
from scipy.odr import *
import math

class Graph:
    def __init__(self, dir_name, number, fold_unfold, count, df_graph=False) -> None:
        self.dir_name = dir_name
        self.number = str(number)
        self.fold_unfold = fold_unfold # u/f
        self.count = count # 1, 2, 3, .....
        self.name = self.dir_name + '/' + self.number + self.fold_unfold + self.count + '.txt'
        self.n_med = 5
        self.df_graph = df_graph
        self.d = 2.0 # nm B-DNA orientation
        self.KBT = 4.11 # pNnm

        # we assume the following elastic parameters & WLC model
        self.lp = 1.35 # persistence length wlc
        self.l = 0.58; # contour length per base wlc
        self.n_aprox = 47 # approximate number of bases of the sequence

    def do_graph(self):
        data = []
        data.append(pd.read_fwf(self.name))
        self.data_frame = pd.concat([j for j in data], ignore_index=True) # I add together all the datasets
        
        '''
            I would like to ask the teacher how to implement this thing
        '''
        # self._rebin()
        
        # print(self.data_frame[['lambda', 'Y_force']])
        # self.data_frame.plot.line(x ='lambda', y='Y_force')
        plt.plot(self.data_frame['lambda'], self.data_frame['Y_force'], label='Y_force')
        plt.ylabel('$f_y$(pN)')
        plt.xlabel('$\lambda$(nm)')
        plt.title(self.name)
        f_F, f_U = self.maximum_f(False)
        # if f_F and f_U:
        #     plt.annotate(f'f_F', xy=(self.data_frame['lambda'][np.where(self.data_frame['Y_force'].values == f_F)[0][0]], f_F),) 
        #     plt.annotate(f'f_U', xy=(self.data_frame['lambda'][np.where(self.data_frame['Y_force'].values == f_U)[0][0]], f_U),) 
        plt.legend()
        plt.show()
        return [f_F, f_U]

    def _check_graph(self, df, df_mean, threshold):
        plt.plot(self.data_frame['lambda'].loc[0:df.size-1], df, label='df_i')
        plt.axhline(y = df_mean, color = 'b', linestyle = 'dashed', label = 'df_mean')    
        plt.axhline(y = threshold, color = 'r', linestyle = 'dashed', label = 'Threshold')    
        plt.axhline(y = -threshold, color = 'r', linestyle = 'dashed')   
        # plt.annotate(f'df_max', xy=(pos1, df_max),) 
        # plt.annotate(f'df_min', xy=(pos2, df_min),) 
        plt.xlabel('$\lambda$')
        plt.ylabel('df')
        plt.title('df')
        plt.legend()
        plt.show()
        plt.close()

    def _func(self, arr, i):
        j = [k for k in arr if k in range(i, i+self.n_med)]
        return j

    def _rebin(self):
        # median filter
        df_empty = pd.DataFrame(columns = self.data_frame.columns) # here I create an empty dataframe, only with same columns of the original one
        df_empty2 = df_empty.copy()
        # y1 = medfilt(self.data_frame['Y_force'], kernel_size=self.n_med)
        for i in range(self.data_frame.shape[0]-self.n_med):
            moment = self.data_frame['Y_force'].loc[i:i+self.n_med-1]
            median_value = moment.median()
            sub_row = self.data_frame.loc[self._func(self.data_frame.index[self.data_frame['Y_force'] == median_value].tolist(), i)[0]]
            df_empty2 = df_empty2.append(sub_row, ignore_index=False)
            # sub_row = self.data_frame.loc[self.data_frame.index[self.data_frame['Y_force'] == y1[i]].tolist()]
            # df_empty2 = df_empty2.append(sub_row, ignore_index=False)
        df_empty2 = df_empty2.reset_index()
        df_empty2 = df_empty2.drop(['index'], axis=1)
        # print(df_empty2)
        # self.data_frame = df_empty2[['lambda', 'Y_force']].copy()
        return df_empty2.copy()
        # print(self.data_frame)

    def maximum_f(self, var=True):

        minimum_f = 3 if self.fold_unfold == 'u' else 2

        f_y_med = self._rebin()
        df_med = np.zeros(f_y_med.shape[0])
        df = np.zeros(f_y_med.shape[0])
        for i in range(df_med.size-2):
            df_med[i] = f_y_med['Y_force'].loc[i+1] - f_y_med['Y_force'].loc[i]
            df[i] = self.data_frame['Y_force'].loc[i+1] - self.data_frame['Y_force'].loc[i]
        df_mean = df.mean()
        df_med_std = df_med.std()
        df_std = df.std()
        print(f"The df mean is {df_mean} and the df std is {df_std}")
        threshold = self.n_med*df_std
        threshold_med = self.n_med*df_std/2
        jump_candidates=[] # candidates for being a jump
        jump_candidates_med=[] # candidates for being jump = difference in force with respect to previous point bigger than threshold
        for i in range(df.size):
            if abs(df[i]) > threshold:
                jump_candidates.append(i)
        for i in range(df_med.size):
            if abs(df_med[i]) > threshold_med:
                jump_candidates_med.append(i)
        print(jump_candidates_med)
        # selecting from the candidates the first one that is over the fmin
        jump = 0 # position of our jump, fullfilling 2 criteria above/fmin & dfy of appropriate sign (depending if unfolding/refolding)
        for k in range(len(jump_candidates_med)):
            if self.fold_unfold == 'u': # unfolding df < 0 
                if df[jump_candidates_med[k]] < 0 and f_y_med['Y_force'].loc[jump_candidates_med[k]] >= minimum_f:
                    jump = jump_candidates_med[k]
                    break
            else:
                if df[jump_candidates_med[k]] > 0 and f_y_med['Y_force'].loc[jump_candidates_med[k]] >= minimum_f:
                    jump = jump_candidates_med[k]
                    break
        if jump == 0:
            print("No jump spotted!")
            self._check_graph(df, df_mean, threshold)
        
        if jump >= 0:
            return self._fit(f_y_med, jump)

        # df_max = df.max()
        # df_min = df.min()
        # TODO
        '''
            This should be better implemented, because this function only find the maximum in the dataframe, above the Threshold, 
            but it is not necessarily the first maximum of whatever.
            Here using also the variable {minimum_force}, there is some work to do...
        '''
        
        # f_F = self.data_frame['Y_force'][np.where(df == df_max)[0][0]+1]
        # f_U = self.data_frame['Y_force'][np.where(df == df_min)[0][0]+1]

        
        # print(f"f_F vale {f_F} e f_U vale {f_U}")

        # if var and self.df_graph:
        self._check_graph(df, df_mean, threshold)
        # if f_F > minimum_f_F and f_U > minimum_f_U:
        #     return [f_F, f_U]

        # else:
        #     return [None, None]



    def _fit(self, f_y_med, jump):
        n_points = 200
        # indpoints = range(jump-2-n_points, jump-2)
        x = f_y_med['lambda'].iloc[jump-2-n_points: jump-2].values.reshape(-1, 1)  # values converts it into a numpy array
        y = f_y_med['Y_force'].iloc[jump-2-n_points: jump-2].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
        linear_regressor = LinearRegression()  # create object for the class
        reg = linear_regressor.fit(x, y)   # perform linear regression
        y_pred = linear_regressor.predict(x)  # make predictions
        plt.plot(x, y, color='blue', label = 'Data')
        plt.plot(x, y_pred, color='red', label = 'Fit')
        plt.xlabel('$\lambda$(nm)')
        plt.ylabel('$f_y$(pN)')
        plt.title('Linear Fit')
        plt.legend()
        plt.show()
        plt.close()
        k_eff = reg.coef_[0][0] # angular coefficient
        lambda0 = reg.intercept_[0] # intercept
        if self.fold_unfold == 'f':
            lu = self.data_frame['lambda'].loc[jump-10:jump].median()
            lf = self.data_frame['lambda'].loc[jump+2:jump+10+2].median()
            f_u = k_eff*lu+lambda0
            f_f = self.data_frame['Y_force'].loc[jump+2:jump+10+2].median()
        else:
            lu = self.data_frame['lambda'].loc[jump+2:jump+10+2].median()
            lf = self.data_frame['lambda'].loc[jump-10:jump].median()
            f_f = k_eff*lf+lambda0
            f_u = self.data_frame['Y_force'].loc[jump+2:jump+10+2].median()
        print(f"Il coefficient k_eff = {k_eff}(pN/nm), la f_F = {f_f}(pN) e la f_U = {f_u}(pN)")

        # Obtaining number of released bases corresponding to this released
        #             Extension of ssDNA

        Df=abs(f_f-f_u)
        if self.fold_unfold == 'u':
            x_d = 1.*self.d*(1./(math.tanh(f_u*self.d/self.KBT))-self.KBT/(f_u*self.d))
            # obtaining released ssDNA
            x_ssDNA = Df/k_eff+x_d
            # nucleotides(i)=contour_WLC(KBT,lp,fu,xssDNA,l*naprox)/l;%the l in the function is helpful to put some boundaries to the obtained number of nucleotides
        else: 
            x_d = 1.*self.d*(1./(math.tanh(f_f*self.d/self.KBT))-self.KBT/(f_f*self.d))
            # obtaining released ssDNA
            x_ssDNA = Df/k_eff+x_d
            # nucleotides(i)=contour_WLC(KBT,lp,ff,xssDNA,l*naprox)/l;%the l in the function is helpful to put some boundaries to the obtained number of nucleotides   
        print(f"Salto avvenuto tra {f_f}(pN) e {f_u}(pN)")
        print(f"x_d = {x_d}, con un x_ssDNA = {x_ssDNA}(nm)")
        return [f_f, f_u]