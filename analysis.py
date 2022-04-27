from files_sorting import Txt
import pandas as pd
import numpy as np

# directory name
dir_name = "Pulling_3bs"
# molecule name is a number in these experimets
number = "16"

file = Txt(dir_name, number)
dataFrame = file.bricolage()
# print(dataFrame)