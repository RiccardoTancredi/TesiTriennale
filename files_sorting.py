import os.path
from click import password_option
import pandas as pd
import numpy as np

class Txt:
    def __init__(self, dir_name, number) -> None:
        self.dir_name = dir_name
        self.number = str(number)
        self.name = self.dir_name + '/' + self.number

    def bricolage(self):
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
        data_frame.reset_index()
        # data_frame.drop(['index'], axis=1)

        # Create the new column lamnda, variable useful later
        data_frame['lambda'] = ((data_frame['A_dist-Y']+data_frame['B_dist-Y']).values)/2

        # Store this dataFrame into a .txt file
        if not os.path.isfile(self.name+'.txt'):
            with open(os.path.join(self.dir_name, self.number+'.txt'),'w') as outfile:
                data_frame.to_string(outfile)

        self._createTxt(data_frame)
        return data_frame

    def _createTxt(self, data_frame):
        # Now I want to create short dataFrame with only folding or unfolding conditions

        # status == 130 => FOLDING
        # status == 131 => UNFOLDING
        data_unfolding = data_frame[data_frame['Status'] == 131]
        data_folding = data_frame[data_frame['Status'] == 130]
        # I want now to create little .txt files with only one folding or unfolding situations:
        
        # strong if here! If this thing has already been made, skip!!!

        file_name = None # number + u/f

        self._createLog(data_frame, file_name) # here I pass the i-th dataFrame in order to be shortly analysed
                                               # these dataframe must be indexed to zero in order to be here!!!
        pass

    def _createLog(self, data_frame, file_name):
        count_in = data_frame.loc[0].at['CycleCount']
        count_fin = data_frame.loc[data_frame.shape[0]-1].at['CycleCount']
        force_x_mean = data_frame['X_force'].mean()
        sigma_force_x = data_frame['X_force'].std()
        force_y_mean = data_frame['Y_force'].mean()
        sigma_force_y = data_frame['Y_force'].std()
        force_z_mean = data_frame['Z_force'].mean()
        sigma_force_z = data_frame['Z_force'].std()
        temp_A = data_frame['A_Temperature'].mean()
        sigma_temp_A = data_frame['A_Temperature'].std()
        temp_B = data_frame['B_Temperature'].mean()
        sigma_temp_B = data_frame['B_Temperature'].std()
        t_in = data_frame.loc[0].at['time(sec)']
        t_fin = data_frame.loc[data_frame.shape[0]-1].at['time(sec)']
        delta_time = t_fin-t_in
        df = pd.DataFrame({"NomeFile": file_name, "count_in": count_in, "count_fin": count_fin, "force_x_mean": force_x_mean, 
        "sigma_force_x": sigma_force_x, "force_y_mean": force_y_mean, "sigma_force_y": sigma_force_y, "force_z_mean": force_z_mean, 
        "sigma_force_z": sigma_force_z, "temp_A": temp_A, "sigma_temp_A": sigma_temp_A, "temp_B": temp_B, "sigma_temp_B": sigma_temp_B,
        "delta_time": delta_time})
        with open(os.path.join(self.dir_name, file_name+'.txt'),'w') as outfile:
            df.to_string(outfile)

        pass