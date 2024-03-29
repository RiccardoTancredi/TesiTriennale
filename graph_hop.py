import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# from sklearn.linear_model import LinearRegression
from scipy.optimize import leastsq, curve_fit
from scipy.integrate import quad
import os.path


class Graph_hop:
    def __init__(self, dir_name, number, number_file) -> None:
        self.dir_name = dir_name
        self.number = str(number)
        self.number_file = str(number_file)
        self.name = self.dir_name + '/' + self.number + '_' + self.number_file 
        self.KBT = 4.11 # pN*nm
        self.d = 2 #nm
        self.P = 1.35 #nm -> persistence length
        d_aa = 0.58 #nm -> distance between consecutive nucleotides
        N = 46 # number of nucleotides
        self.L = N*d_aa # nm
        self.nome_txt = self.dir_name + '/' + self.number + 'output.txt' 
        self.output = self.name # We will copy this string into an output txt file
        
    def do_graph(self, max_time=None, space=0):
        data = []
        data.append(pd.read_fwf(self.name+'.txt', colspecs = [(0, 9), (9, 18), (18, 28+space), (28+space, 38+space), (38+space, -1)]))
        self.data_frame = pd.concat([j for j in data], ignore_index=True) # I add together all the datasets
        '''
            I would like to ask the teacher how to implement this thing
        '''
        # self._rebin()
        self.data_frame['time(sec)'] = self.data_frame['time(sec)'].sub(self.data_frame['time(sec)'].loc[0]) 

        if max_time:
            self.data_frame = self.data_frame.loc[self.data_frame['time(sec)']<max_time]

        self.graph()
        self.histogram()

        return self.data_frame


    def _doublegaussian(self, params, x):
        # params is a vector of the parameters:
        # params = [f_F, sigma_F, w_F, f_U, sigma_U, w_U]
        (c1, mu1, sigma1, c2, mu2, sigma2) = params
        res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
        return res
        # return B[2]/np.sqrt(2*np.pi*B[1])*np.exp(((x-B[0])/(2*B[1]))**2) + B[5]/np.sqrt(2*np.pi*B[4])*np.exp(((x-B[3])/(2*B[4]))**2)

    def _double_gaussian_fit(self, params, errors=False):
        if not errors:
            fitting = self._doublegaussian(params, self.bin)
            return (fitting - self.values_histogram_bins_proc)
        else:
            self.doublegaussian_1 = np.vectorize(self._doublegaussian, excluded=['params'])
            parameters = np.copy(params)
            fitting = self.doublegaussian_1(params=parameters, x=self.bin)
            return (fitting - self.values_histogram_bins_proc)


    def fit(self, guess:list):
        fitting, pcov, infodict, errmsg, success = leastsq(self._double_gaussian_fit, guess, full_output=1, epsfcn=0.0001) # pcov, infodict, errmsg, success
        if pcov is not None:
            s_sq = (self._double_gaussian_fit(fitting.tolist(), True)**2).sum()/(self.values_histogram_bins_proc.shape[0]-len(guess))
            pcov = pcov * s_sq
        else:
            pcov = np.inf
        error = []
        for i in range(len(fitting)):
            try:
                error.append(np.abs(pcov[i][i])**0.5)
            except:
                error.append(0.00)
        err_leastsq = np.array(error)
        prob, err_prob = self._fit_plot(fitting, err_leastsq)
        return fitting, err_leastsq, prob, err_prob
    
    
    def graph(self):
        df_mean = self.data_frame['Y_force'].mean()
        df_std = self.data_frame['Y_force'].std()
        print(f"f media vale = {df_mean}, con deviazione standard = {df_std}")
        plt.plot(self.data_frame['time(sec)'], self.data_frame['Y_force'], label='Data')
        plt.axhline(y = df_mean, color = 'b', linestyle = 'dashed', label = '$\mu$')    
        plt.axhline(y = df_mean+3*df_std, color = 'r', linestyle = 'dashed', label = '$\mu\pm3\sigma$')   
        plt.axhline(y = df_mean-3*df_std, color = 'r', linestyle = 'dashed')   
        plt.ylabel('$f_y\:[pN]$', fontsize=15)
        plt.xlabel('$t\:[s]$', fontsize=15)
        # plt.title(self.name)
        plt.legend()
        plt.show()
        self.output += f"f media vale = {df_mean}, con deviazione standard = {df_std}\n"

    def histogram(self):
        rice = int(6*np.cbrt(self.data_frame.shape[0]))
        # scott = int(3.49*self.data_frame['Y_force'].std()/np.cbrt(self.data_frame.shape[0]))
        plt.ylabel('$f_y\:[pN]$', fontsize=15)
        plt.xlabel('$p(f)\:[1/pN]$', fontsize=15)
        # plt.title(self.name+ ': Force Histogram')
        # self.data_frame['Y_force'].hist(grid=False, bins=rice)
        self.values_histogram_bins, bins, patches = plt.hist(self.data_frame['Y_force'], density=True, bins=rice, orientation='horizontal', label='Data', stacked=True) # y
        self.bin = [(bins[i+1] + bins[i])/2 for i in range(len(bins)-1)] # x
        self.values_histogram_bins_proc = np.copy(self.values_histogram_bins)
        self.values_histogram_bins_proc[self.values_histogram_bins_proc < 0.05] = 0.0
        plt.legend()
        plt.show()
    
    def _fit_plot(self, fitting, err_leastsq):
        rice = int(6*np.cbrt(self.data_frame.shape[0]))
        plt.hist(self.data_frame['Y_force'], density=True, bins=rice, orientation='horizontal', label='Data', stacked=True)
        plt.plot(self._doublegaussian(fitting, self.bin), self.bin, c='r', label='Fit')
        plt.axhline(y = fitting[1], color = 'g', linestyle = 'dashed', label = '$f_U$')
        plt.axhline(y = fitting[4], color = 'y', linestyle = 'dashed', label = '$f_N$')    
        plt.ylabel('$f_y\:[pN]$', fontsize=15)
        plt.xlabel('$p(f)\:[1/pN]$', fontsize=15)
        # plt.title(self.name+ ': Force Histogram + Fit')
        plt.legend()
        plt.show()
        print(f"c_1 = {fitting[0]}, mu_1 = {fitting[1]}, sigma_1 = {fitting[2]}")
        print(f"c_2 = {fitting[3]}, mu_2 = {fitting[4]}, sigma_2 = {fitting[5]}")
        print(f"sigma_c_1 = {err_leastsq[0]}, sigma_mu_1 = {err_leastsq[1]}, sigma_sigma_1 = {err_leastsq[2]}")
        print(f"sigma_c_2 = {err_leastsq[3]}, sigma_mu_2 = {err_leastsq[4]}, sigma_sigma_2 = {err_leastsq[5]}")

        self.output += f"c_1 = {fitting[0]}, mu_1 = {fitting[1]}, sigma_1 = {fitting[2]}\n" + f"c_2 = {fitting[3]}, mu_2 = {fitting[4]}, sigma_2 = {fitting[5]}\n" + \
            f"sigma_c_1 = {err_leastsq[0]}, sigma_mu_1 = {err_leastsq[1]}, sigma_sigma_1 = {err_leastsq[2]}\n" + f"sigma_c_2 = {err_leastsq[3]}, sigma_mu_2 = {err_leastsq[4]}, sigma_sigma_2 = {err_leastsq[5]}\n"

        w_N = fitting[0]*np.sqrt(2*np.pi*fitting[2]**2)
        w_U = fitting[3]*np.sqrt(2*np.pi*fitting[5]**2)
        sigma_w_N = w_U*np.sqrt((err_leastsq[0]/fitting[0])**2 + (err_leastsq[2]/(2*fitting[2]))**2)
        sigma_w_U = w_N*np.sqrt((err_leastsq[3]/fitting[3])**2 + (err_leastsq[5]/(2*fitting[5]))**2)
        print(f"w_U = {w_U}, sigma_w_U = {sigma_w_U}")
        print(f"w_N = {w_N}, sigma_w_N = {sigma_w_N}")
        self.output += f"w_U = {w_U}, sigma_w_U = {sigma_w_U}\n" + f"w_N = {w_N}, sigma_w_N = {sigma_w_N}\n"
        return [w_U, w_N], [sigma_w_U, sigma_w_N]


    # def _prova(self):
    #     plt.plot(self.bin, self.values_histogram_bins)
    #     plt.show()

    def subplots(self, fitting, Markov=True, n_points_fig=None, X=None):
        df_mean = self.data_frame['Y_force'].mean()
        df_std = self.data_frame['Y_force'].std()
        # Setting up the plot surface
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(nrows=1, ncols=4)
        # First axes
        ax0 = fig.add_subplot(gs[0, :2])
        n_points = self.data_frame.shape[0] if not n_points_fig else n_points_fig
        ax0.plot(self.data_frame['time(sec)'][:n_points], self.data_frame['Y_force'][:n_points], label='Y_force')
        ax0.axhline(y = df_mean, color = 'b', linestyle = 'dashed', label = '$\mu$')    
        ax0.axhline(y = df_mean+3*df_std, color = 'r', linestyle = 'dashed', label = '$\mu\pm3\sigma$')   
        ax0.axhline(y = df_mean-3*df_std, color = 'r', linestyle = 'dashed')
        states_hhm = self.hmm(fitting, X)
        if Markov:
            ax0.plot(self.data_frame['time(sec)'][:n_points], states_hhm[:n_points], color = 'c', label='HMM')
        ax0.set_ylabel('$f_y\:[pN]$')
        ax0.set_xlabel('$t\:[s]$')
        # ax0.set_title(self.name)
        ax0.legend()
        # Second axes
        rice = int(6*np.cbrt(self.data_frame.shape[0]))
        ax1 = fig.add_subplot(gs[0, 2:4])
        rice = int(6*np.cbrt(self.data_frame.shape[0]))
        ax1.hist(self.data_frame['Y_force'], density=True, bins=rice, orientation='horizontal', label='Data', stacked=True)
        ax1.plot(self._doublegaussian(fitting, self.bin), self.bin, c='r', label='Fit')
        ax1.axhline(y = fitting[1], color = 'g', linestyle = 'dashed', label = '$\mu_1$')
        ax1.axhline(y = fitting[4], color = 'y', linestyle = 'dashed', label = '$\mu_2$')    
        # ax1.set_ylabel('$f_y$(pN)')
        if n_points_fig:
            ax1.set_yticks([])
        ax1.set_xlabel('$p(f)\:[1/pN]$')
        # ax1.set_title(self.name+ ': Force Histogram + Fit')
        ax1.legend()
        # plt.subplots_adjust(wspace=0.08,)
        plt.show()

    def _linear(self, x, m, q):
        res =   m*x+q
        return res
        
    def _linear_fit(self, x, y, params, sigma_y=None):
        linear = np.vectorize(self._linear,  excluded=['m', 'q'])
        absol = True if sigma_y else False
        popt, pcov = curve_fit(linear, x, y, params, sigma_y, absolute_sigma=absol)
        sigmas = np.sqrt(np.diag(pcov))
        return popt, sigmas

    def deltaG(self, w_U, w_N, forces, sigma_w_U, sigma_w_N, par=None, particolare=None, zoom_out=None):
        # linear fit: k_B T log(w_U/w_N) = (f-f_c)*x_NU = m*f + q, m = x_NU, q = f_c*x_NU 
        y = [self.KBT*np.log(w_U[i]/w_N[i]) for i in range(len(w_U))]
        sigma_y = [self.KBT*np.sqrt((sigma_w_U[i]/w_U[i])**2+(sigma_w_N[i]/w_N[i])**2) for i in range(len(sigma_w_N))]
        x = forces
        ### New method
        guess = [-12.5, 59.6] if not par else par
        (m, q), (sigma_m, sigma_q) = self._linear_fit(x, y, guess, sigma_y)
        # linear_regressor = LinearRegression()  # create object for the class
        # reg = linear_regressor.fit(x, y)   # perform linear regression
        # y_pred = linear_regressor.predict(x)  # make predictions
        linear = np.vectorize(self._linear,  excluded=['m', 'q'])
        y_pred = linear(x=x, m=m, q=q)
        y = np.array(y).reshape(-1, 1)
        x = np.array(x).reshape(-1, 1)
        plt.ylabel('$ln(w_U\:/\:w_N)$', fontsize=15)
        plt.xlabel('$f\:[pN]$', fontsize=15)
        plt.errorbar(x, y, sigma_y, fmt = 'o', color='blue', label = 'Data')
        plt.plot(x, y_pred, color='red', label = 'Fit')
        # plt.title('$w_U\:/\:w_N \:- Linear\: Fit$')
        if particolare:
            plt.axvline(x = 3, ymin=np.exp(-15), ymax=np.exp(0.2), color = 'g', linestyle = '--')
        if zoom_out:
            plt.xlim(2.5, 5.5)
            plt.ylim(-15, 20)
        plt.legend()
        plt.show()

        x_NU = m
        self.x_NU = x_NU
        sigma_x_nU = sigma_m
        self.sigma_x_NU = sigma_x_nU
        f_c = -q/m
        sigma_f_c = f_c*np.sqrt((sigma_q/q)**2+(sigma_m/m)**2)
        DeltaG_NU = -q
        sigma_DeltaG_NU = sigma_q
        print(f"La forza di coesistenza vale f_c = {f_c}, con sigma = {sigma_f_c}")
        print(f"La differenza di lunghezza tra lo stato foldend e unfolded e' x_NU = {x_NU}, con sigma = {sigma_x_nU}")
        print(f"La differenza di energia libera DeltaG_NU = {DeltaG_NU}, con sigma = {sigma_DeltaG_NU}")
        self.output += "\n Linear Fit \n"
        self.output += f"La forza di coesistenza vale f_c = {f_c}, con sigma = {sigma_f_c}\n" + f"La differenza di lunghezza tra lo stato foldend e unfolded e' x_NU = {x_NU}, con sigma = {sigma_x_nU}\n" \
            f"La differenza di energia libera DeltaG_NU = {DeltaG_NU}, con sigma = {sigma_DeltaG_NU}\n"
        return (x_NU, sigma_x_nU), (f_c, sigma_f_c), (DeltaG_NU, sigma_DeltaG_NU)
        
    def x_d(self, f):
        return self.d*(1./np.tanh((f*self.d)/self.KBT) - self.KBT/(f*self.d)) # nm

    def f_WLC(self, x):
        return self.KBT/self.P * (1./(4*(1-x/self.L)**2) - 1/4 + x/self.L) # pN

    def G0(self, fc):
        # f_c = coexistence force
        f_c, sigma_f_c = fc
        x_fc = self.x_WLC_f(f_c) # extension at f_c, by inverting the formula for the force in the WLC model (see below)
        # calculate integral using basic scipy method
        G0_delta = x_fc*f_c-quad(self.f_WLC, 0, x_fc)[0] - quad(self.x_d, 0, f_c)[0] # check if these integrals are correct
        # calcolo incertezza
        f_c += sigma_f_c
        x_fc = self.x_WLC_f(f_c)
        G0_piu_sigma = x_fc*f_c-quad(self.f_WLC, 0, x_fc)[0] - quad(self.x_d, 0, f_c)[0]
        f_c -= 2*sigma_f_c
        x_fc = self.x_WLC_f(f_c)
        G0_meno_sigma = x_fc*f_c-quad(self.f_WLC, 0, x_fc)[0] - quad(self.x_d, 0, f_c)[0]
        sigma_G0_delta = (G0_piu_sigma-G0_meno_sigma)/np.sqrt(24) # distribuzione triangolare
        print(f"DeltaG0 = {G0_delta}, con sigma = {sigma_G0_delta}")
        self.output += f"DeltaG0 = {G0_delta}, con sigma = {sigma_G0_delta}\n"
        return G0_delta, sigma_G0_delta


    # Hidden Markov Model - 2 hidden states
    def hmm(self, fitting, X = None):
        (c1, mu1, sigma1, c2, mu2, sigma2) = fitting
        states = []
        if X:
            for x in X:
                if x == 0:
                    states.append(mu1)
                else:
                    states.append(mu2)
        else:
            for mis in self.data_frame['Y_force']:
                prob_1 = self._gaussian(mis, [mu1, sigma1])
                prob_2 = self._gaussian(mis, [mu2, sigma2])
                if prob_1>prob_2:
                    states.append(mu1)
                else:
                    states.append(mu2)
        return np.array(states)


    def _gaussian(self, x, par):
        mu, sigma = par
        return np.exp(- (x - mu)**2.0 / (2.0 * sigma**2.0) )/np.sqrt(2*np.pi*sigma**2.)


    def hmm_analysis(self, fitting):
        (c1, mu1, sigma1, c2, mu2, sigma2) = fitting
        states = self.hmm(fitting)
        # Time selection data equals to every dataset
        native = len([i for i in states if i == mu1])*1e-3 # up force # [t_min:t_max]
        unfolded = len([j for j in states if j == mu2])*1e-3 # or faster: len(states[t_min:t_max]) - native
        print(f"La molecola si trova {native} sec nello stato nativo e {unfolded} sec nello stato unfolded")
        self.output += f"La molecola si trova {native} sec nello stato nativo e {unfolded} sec nello stato unfolded\n"
        return native, unfolded

    def residence_time(self, native_time, unfolded_time, forces, sigma_forces, par1=None, par2=None):
        # grafico tempi vs forze_medie di esistenza stato folded e unfolded    
        linear = np.vectorize(self._linear,  excluded=['m', 'q'])
        guess1 = [-10, 100] if not par1 else par1
        guess2 = [10, -50] if not par2 else par2
        (m_1, q_1), (sigma_m_1, sigma_q_1) = self._linear_fit(np.log(native_time), forces, guess1, sigma_forces) # , sigma_y
        (m_2, q_2), (sigma_m_2, sigma_q_2) = self._linear_fit(np.log(unfolded_time), forces, guess2, sigma_forces) # , sigma_y
        x = np.linspace(min(np.log(native_time+unfolded_time)), max(np.log(native_time+unfolded_time)), 1000)
        y_pred1 = linear(x=x, m=m_1, q=q_1)
        y_pred2 = linear(x=x, m=m_2, q=q_2)
        f_c = (q_2*m_1-q_1*m_2)/(m_1-m_2)
        sigma_f_c = np.sqrt(((m_2*sigma_q_1)/(m_2-m_1))**2+((m_1*sigma_q_2)/(m_2-m_1))**2+((m_1*sigma_m_2*(q_2-q_1))/((m_2-m_1)**2))**2+((m_2*sigma_m_1*(q_1-q_2))/((m_2-m_1)**2))**2)
        t_c = np.exp((q_2-q_1)/(m_1-m_2))
        sigma_t_c = t_c*np.sqrt((sigma_q_2/(m_1-m_2))**2+(sigma_q_1/(m_1-m_2))**2+((q_2-q_1)*sigma_m_1/((m_2-m_1)**2))**2+((sigma_m_2*(q_1-q_2))/((m_2-m_1)**2))**2)
        plt.errorbar(np.log(native_time), forces, sigma_forces, fmt = 'o', c='r', label='$t_N$')
        plt.plot(x, y_pred1, c='r', label='Fit')
        plt.errorbar(np.log(unfolded_time), forces, sigma_forces, fmt = 'o', c='b', label='$t_U$')
        plt.plot(x, y_pred2, c='b', label='Fit')
        x_oriz = np.linspace(min(np.log(native_time+unfolded_time)), np.log(t_c), 100)
        y_vert = np.linspace(min(y_pred2.tolist()+y_pred1.tolist()), f_c, 100)
        plt.plot(x_oriz, [f_c]*x_oriz.shape[0], color = 'g', linestyle = 'dashed', label = '$(t,\: f_c)$')
        plt.plot([np.log(t_c)]*y_vert.shape[0], y_vert, color = 'g', linestyle = 'dashed')
        plt.xlabel('$log(t) \: [s]$')
        plt.ylabel('$\overline{f} \:[pN]$')
        # plt.title('Log Residence Time')
        plt.legend()
        plt.show()
        beta = (1./m_2-1./m_1)/self.x_NU
        DeltaGNU = (-q_1/m_1+q_2/m_2)/beta
        sigma_DeltaGNU = np.sqrt((m_1*self.x_NU*sigma_q_2/(m_1-m_2))**2+(m_2*sigma_q_1*self.x_NU/(m_1-m_2))**2+(m_2*(q_1-q_2)*self.x_NU*sigma_m_1/(m_1-m_2)**2)**2+(m_1*self.x_NU*sigma_m_2*(q_2-q_1)/(m_1-m_2)**2)**2+(DeltaGNU*self.sigma_x_NU/self.x_NU)**2)
        print(f"Stimiamo i parametri del fit lineare: m1 = {m_1}, con incertezza = {sigma_m_1}, \n q1 = {q_1}, con incertezza = {sigma_q_1}")
        print(f"Stimiamo i parametri del fit lineare: m2 = {m_2}, con incertezza = {sigma_m_2}, \n q2 = {q_2}, con incertezza = {sigma_q_2}")
        print(f"La forza di coesistenza qui vale: fc = {f_c}, con incertezza = {sigma_f_c}")
        print(f"Il tempo medi di residenza vale: tc = {t_c}, con incertezza = {sigma_t_c}")
        print(f"Stimiamo un nuovo DeltaG_NU = {DeltaGNU}, con incertezza = {sigma_DeltaGNU}")
        self.output += "\n Log Residence Time Fit \n"
        self.output += f"Stimiamo i parametri del fit lineare: m1 = {m_1}, con incertezza = {sigma_m_1}, \n q1 = {q_1}, con incertezza = {sigma_q_1}\n" \
            + f"Stimiamo i parametri del fit lineare: m2 = {m_2}, con incertezza = {sigma_m_2}, \n q2 = {q_2}, con incertezza = {sigma_q_2}\n" \
                + f"La forza di coesistenza qui vale: fc = {f_c}, con incertezza = {sigma_f_c}\n" + f"Il tempo medi di residenza vale: tc = {t_c}, con incertezza = {sigma_t_c}\n" \
                    + f"Stimiamo un nuovo DeltaG_NU = {DeltaGNU}, con incertezza = {sigma_DeltaGNU}\n"
        return (m_1, sigma_m_1), (q_1, sigma_q_1), (m_2, sigma_m_2), (q_2, sigma_q_2), (f_c, sigma_f_c), (t_c, sigma_t_c), (DeltaGNU, sigma_DeltaGNU)
       

    # Inverse function of f(x) from WLC model
    def x_WLC_f(self, f):
        fnorm = ((4*self.P)/self.KBT)*f
        a2 = (1/4)*(-9-fnorm)
        a1 = (3/2)+(1/2)*fnorm
        a0 = -fnorm/4
        
        R = (9*a1*a2-27*a0-2*a2**3)/54.
        Q = (3*a1-a2**2)/9.
        
        D = Q**3+R**2
        
        if D > 0:
            # In this case, there is only one real root, given by "out" below
            S = np.cbrt(R+np.sqrt(D))
            T = np.cbrt(R-np.sqrt(D))
            out = (-1/3)*a2+S+T
        elif D < 0:
            # In this case there 3 real distinct solutions, given by out1,
            # out2, out3 below. The one that interests us is that in the
            # inerval [0,1]. It is seen ("empirically") that is always the
            # second one in the list below [there is perhaps more to search here]
            
            theta = np.arccos(R/np.sqrt(-Q**3))
            # out1 = 2*np.sqrt(-Q)*np.cos(theta/3)-(1/3)*a2;
            out2 = 2*np.sqrt(-Q)*np.cos((theta+2*np.pi)/3)-(1/3)*a2
            # out3 = 2*np.sqrt(-Q)*np.cos((theta+4*np.pi)/3)-(1/3)*a2
            
            # We implement the following check just to be sure out2 is the good root 
            # (in case this "empirical" truth turns out to stop working) 
            try:
                out2 < 0 or out2 > 1
            except:    
                print('The default root doesn"t seem the be good one - you may want to check if the others lie in the interval [0,1]')
            else:
                out = out2
        else:
            # In theory we always go from D>0 to D<0 by passing to a D=0
            # boundary, where we have two real roots (and where the formulas
            # above change again slightly). In practice, however, due to round-off errors,
            # it seems we never hit this boundary but always pass "through" it 
            # This D=0 scenario could still be implemented if needed, though.
            print('#ToDo')

        z = out
        return z*self.L   


    def write_on_txt(self):
        status = "r" if os.path.isfile(self.nome_txt) else "a"
        f = open(self.nome_txt, status)
        f.write(self.output)
        f.close()