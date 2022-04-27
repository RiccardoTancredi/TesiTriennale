import os.path
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
        for i in alphabet:
            moment_name = str(self.name+i+'.txt')
            if os.path.isfile(moment_name):
                data.append(pd.read_csv(moment_name, sep = "\t"))
            else:
                break

        data_frame = pd.concat([j for j in data], ignore_index=True)
        data_frame.rename(columns = {'#CycleCount/n':'CycleCount'}, inplace = True)
        data_frame.drop(data_frame[data_frame['CycleCount'] == "#End Write"].index, inplace = True)
        data_frame.drop(data_frame[data_frame['CycleCount'] == "#Skipped"].index, inplace = True)

        return data_frame
