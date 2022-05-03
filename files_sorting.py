import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Txt:
    def __init__(self, dir_name, number) -> None:
        self.dir_name = dir_name
        self.number = str(number)
        self.name = self.dir_name + '/' + self.number
        self.first_time = False # if the file is open for the first time use method _removeBegin

    def _removeBegin(self):
        if self.first_time:
            a_file = open(self.name+'A.txt', "r")
            lines = a_file.readlines()
            a_file.close()
            if "#Begin Write:" in lines[0]:
                del lines[0]
            new_file = open(self.name+'A.txt', "w+")
            for line in lines:
                new_file.write(line)
            new_file.close()

    def bricolage(self):
        self._removeBegin()
        alphabet = ['A','B','C','D','E','F','G','H','I','L','M','N','O','P','Q','R','S','T','U','V','Z','J','K','X','Y','W'] 
        data = []
        # Here I check if there are files for the same molecul and I open them and put them into the same dataFrame
        for i in alphabet:
            moment_name = str(self.name+i+'.txt')
            if os.path.isfile(moment_name):
                data.append(pd.read_csv(moment_name, sep = "\t"))
            else:
                break
        data_frame = pd.concat([j for j in data], ignore_index=True) # I add together all the datasets
        data_frame.rename(columns = {'#CycleCount/n':'CycleCount'}, inplace = True) 
        # Remove the keywords: #EndWrite and #Skipped
        data_frame.drop(data_frame[data_frame['CycleCount'] == "#End Write"].index, inplace = True)
        data_frame.drop(data_frame[data_frame['CycleCount'] == "#Skipped"].index, inplace = True)
        data_frame = data_frame.reset_index()
        data_frame = data_frame.drop(['index'], axis=1)

        # Create the new column lamnda, variable useful later
        data_frame['lambda'] = ((data_frame['A_dist-Y']+data_frame['B_dist-Y']).values)/2

        '''
            Store this dataFrame into a .txt file
        '''

        if not os.path.isfile(self.name+'.txt'):
            with open(os.path.join(self.dir_name, self.number+'.txt'),'w') as outfile:
                data_frame.to_string(outfile, index=False)
        data_frame['Y_force'] = data_frame['Y_force'].abs()
        self._createTxt(data_frame)
        return data_frame

    def _createTxt(self, data_frame):
        
        # Now I want to create short dataFrame with only folding or unfolding conditions creating little .txt files with only one folding or unfolding situations:
        if not os.path.isfile(self.name+'f1'+'.txt'): # here I check if the first type of file exist, which is n_molecul_u1.txt, where "u" stands for unfolding
            # status == 130 => FOLDING
            # status == 131 => UNFOLDING
            df_empty = pd.DataFrame(columns = data_frame.columns) # here I create an empty dataframe, only with same columns of the original one
            df_empty2 = df_empty.copy() # better to work with a copy
            df_empty3 = df_empty.copy()

            lista_folding = data_frame.index[data_frame['Status'] == 130].tolist()
            lista_unfolding = data_frame.index[data_frame['Status'] == 131].tolist()
            lista_folding.append(0) 
            lista_unfolding.append(0)
            j = 0


            '''
                With these two for loops we create files with single folding and unfolding states
            '''
            for i in range(len(lista_folding)-1):
                if lista_folding[i+1] - lista_folding[i] == 1:
                    df_empty2 = df_empty2.append(data_frame.loc[lista_folding[i]], ignore_index=True)
                else:
                    df_empty2 = df_empty2.append(data_frame.loc[lista_folding[i]], ignore_index=True)
                    with open(os.path.join(self.dir_name, self.number+'f'+str(j+1)+'.txt'), 'w') as outfile:
                        df_empty2[['X_force', 'Z_force', 'lambda', 'Y_force', 'time(sec)']].to_string(outfile, index=False)
                    j += 1
                    self._createLog(df_empty2, j, 0)
                    df_empty2 = df_empty.iloc[0:0].copy()  # empty the dataframe
                    
            j = 0
            for i in range(len(lista_unfolding)-1):
                if lista_unfolding[i+1] - lista_unfolding[i] == 1:
                    df_empty3 = df_empty3.append(data_frame.loc[lista_unfolding[i]], ignore_index=True)
                else:
                    df_empty3 = df_empty3.append(data_frame.loc[lista_unfolding[i]], ignore_index=True)
                    with open(os.path.join(self.dir_name, self.number+'u'+str(j+1)+'.txt'), 'w') as outfile:
                        df_empty3[['X_force', 'Z_force', 'lambda', 'Y_force',  'time(sec)']].to_string(outfile, index=False)
                    j += 1
                    self._createLog(df_empty3, j, 1)
                    df_empty3 = df_empty.iloc[0:0].copy()  # empty the dataframe
        
        self._controlplot()

    def _createLog(self, data_frame, file_name, fold_unfold):
        
        # minimum = min(len([name for name in os.listdir("Pulling_3bs") if os.path.isfile("Pulling_3bs"+'/'+name) if 'u' in name]), len([name for name in os.listdir("Pulling_3bs") if os.path.isfile("Pulling_3bs"+'/'+name) if 'f' in name]))
        
        # data = []
        moment_name = self.name
        if fold_unfold == 0:
            moment_name += 'f'+str(file_name)+'.txt'
        else:
            moment_name += 'u'+str(file_name)+'.txt'
        # data.append(pd.read_fwf(moment_name))
        # data_frame = pd.concat([j for j in data], ignore_index=True) # I add together all the datasets
        count_in = [data_frame.loc[0].at['CycleCount']]
        count_fin = [data_frame.loc[data_frame.shape[0]-1].at['CycleCount']]
        force_x_mean = [data_frame['X_force'].mean()]
        sigma_force_x = [data_frame['X_force'].std()]
        force_y_mean = [data_frame['Y_force'].mean()]
        sigma_force_y = [data_frame['Y_force'].std()]
        force_z_mean = [data_frame['Z_force'].mean()]
        sigma_force_z = [data_frame['Z_force'].std()]
        temp_A = [data_frame['A_Temperature'].mean()]
        sigma_temp_A = [data_frame['A_Temperature'].std()]
        temp_B = [data_frame['B_Temperature'].mean()]
        sigma_temp_B = [data_frame['B_Temperature'].std()]
        t_in = [data_frame.loc[0].at['time(sec)']]
        t_fin = [data_frame.loc[data_frame.shape[0]-1].at['time(sec)']]
        delta_time = [t_fin[0]-t_in[0]]
        df = pd.DataFrame({"NomeFile": moment_name, "count_in": count_in, "count_fin": count_fin, "force_x_mean": force_x_mean, 
        "sigma_force_x": sigma_force_x, "force_y_mean": force_y_mean, "sigma_force_y": sigma_force_y, "force_z_mean": force_z_mean, 
        "sigma_force_z": sigma_force_z, "temp_A": temp_A, "sigma_temp_A": sigma_temp_A, "temp_B": temp_B, "sigma_temp_B": sigma_temp_B,
        "delta_time": delta_time})
        with open(os.path.join(self.dir_name, 'Log'+self.number+'.txt'),'a') as outfile:
            df.to_string(outfile, index=False)



    def _controlplot(self):
        data = []
        moment_name = self.dir_name+'/'+'Log'+self.number+'.txt'
        data.append(pd.read_fwf(moment_name))
        data_frame = pd.concat([j for j in data], ignore_index=True) # I add together all the datasets

        #Graphic 
        x = range(data_frame.shape[0])
        plt.plot(x, data_frame['force_y_mean'], label='Force Y')
        plt.plot(x, data_frame['force_x_mean'], label='Force X')
        plt.plot(x, data_frame['force_z_mean'], label='Force Z')
        plt.xlabel('#Steps')
        plt.ylabel('f')
        plt.title('Log'+self.number)
        plt.legend()
        plt.show()