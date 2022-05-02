import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Txt_hop:
    def __init__(self, dir_name, number) -> None:
        self.dir_name = dir_name
        self.number = str(number)
        self.name = self.dir_name + '/' + self.number + 'hop'
        self.first_time_molecule = False # if the file is open for the first time use method _removeBegin

    def _removeBegin(self):
        if self.first_time_molecule:
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
        data_frame['CycleCount'] = pd.to_numeric(data_frame['CycleCount'])
        data_frame = self._freq(data_frame)
        data_frame = data_frame.reset_index()
        data_frame = data_frame.drop(['index'], axis=1)
        data_frame = data_frame.drop(['level_0'], axis=1)
        # Create the new column lamnda, variable useful later
        data_frame['lambda'] = ((data_frame['A_dist-Y']+data_frame['B_dist-Y']).values)/2

        '''
            Store this dataFrame into a .txt file
        '''
        data_frame['Y_force'] = data_frame['Y_force'].abs()
        if not os.path.isfile(self.name+'.txt'):
            with open(os.path.join(self.dir_name, self.number+'.txt'),'w') as outfile:
                data_frame[['X_force', 'Z_force', 'lambda', 'Y_force', 'time(sec)']].to_string(outfile, index=False)
        
        return data_frame

    def _freq(self, data_frame):
        # here we want to take only rows with f = 100 kHz
        data_frame = data_frame[data_frame['CycleCount'].diff() == 1]
        data_frame = data_frame.reset_index()
        return data_frame