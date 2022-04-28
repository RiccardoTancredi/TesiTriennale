# importing the modules
import pandas as pd
import numpy as np

# creating a DataFrame
ODI_runs = {'name': ['Tendulkar', 'Sangakkara', 'Ponting',
					'Jayasurya', 'Jayawardene', 'Kohli',
					'Haq', 'Kallis', 'Ganguly', 'Dravid'],
			'runs': [18426, 14234, 13704, 13430, 12650,
					11867, 11739, 11579, 11363, 10889]}
df = pd.DataFrame(ODI_runs)

# displaying the original DataFrame
print("Original DataFrame :")
print(df)

# dropping the 0th and the 1st index
df = df.drop([2, 3])

# displaying the altered DataFrame
print("DataFrame after removing the 0th and 1st row")
print(df)

# resetting the DataFrame index
df = df.reset_index()

# displaying the DataFrame with new index
print("Dataframe after resetting the index")
print(df)

df = df.drop(['index'], axis=1)
print(df)

print(df.loc[df.shape[0]-1])
