# # # importing the modules
# import pandas as pd
# import numpy as np
# import os.path

# # # creating a DataFrame
# ODI_runs = {'name': ['Tendulkar', 'Sangakkara', 'Ponting',
# 					'Jayasurya', 'Jayawardene', 'Kohli',
# 					'Haq', 'Kallis', 'Ganguly'],
# 			'runs': [-18426, 14234, 13704, 13430, -12650,
# 					11867, 11739, -11579, 11363],
# 			'status': [30, 130, 131, 31, 130, 130, 131, 131, 131]}
# df = pd.DataFrame(ODI_runs)

# # # # displaying the original DataFrame
# # # print("Original DataFrame :")
# # # print(df)

# # # # dropping the 0th and the 1st index
# # # df = df.drop([2, 3])

# # # # displaying the altered DataFrame
# # # print("DataFrame after removing the 0th and 1st row")
# # # print(df)

# # # # resetting the DataFrame index
# # # df = df.reset_index()

# # # # displaying the DataFrame with new index
# # # print("Dataframe after resetting the index")
# # # print(df)

# # # df = df.drop(['index'], axis=1)
# # # print(df)

# # # print(df.loc[df.shape[0]-1])

# # if not os.path.isfile('Pulling_3bs'+'/'+'16''.txt'):
# # 	print("Ciao")


# # df_empty = pd.DataFrame({'A' : []})

# # # for i in range(df.shape[0]):
# # 	# df_empty = df.loc[df['status'] == 130]

# # # print(df_empty)
# # # print(df_empty.iloc[1].index)

# # lista = df.index[df['status'] == 130].tolist()
# # # print(lista)
# # # print(df.loc[0])
# # sub_lista = []
# # # while len(lista) != 0:
# # j = 0
# # # for i in range(len(lista)-1):
# # # 	if lista[i+1] - lista[i] == 1:
# # # 		df_empty.append(df.loc[lista[i]], ignore_index=True)
# # # 	else:
# # # 		df_empty.append(df.loc[lista[i]], ignore_index=True)
# # # 		df_empty.reset_index()
# # # 		with open(os.path.join("Pulling_3bs", 'Prova'+str(j)+'.txt'),'w') as outfile:
# # # 			df_empty.to_string(outfile)
# # # 		j += 1
# # # 		df_empty.iloc[0:0] # svuota il dataframe

# # # print(df)
# # file_name = 'Pulling_3bs' + '/16' + 'log.txt'
# # # print(file_name)
# # # print(os.path.exists(file_name))
# # # C:\Users\ricta\Documents\TesiTriennale\Pulling_3bs\16log,txt
# # # print( len([name for name in os.listdir('Pulling_3bs') if os.path.isfile('Pulling_3bs/'+name)]))

# # # print(os.path.isfile("Pulling_3bs"+'/'+'16' + 'log.txt'))
# # # print(df)
# # # print(df.loc[1:3])
# # print((df.iloc[[df.index[df['name']=='Sangakkara']][0]]))
# # # print(df.index[df['name']=='Sangakkara'])



# # data = []
# # data.append(pd.read_csv('Pulling_3bs/16u86.txt', delimiter = "\t"))
# # data_frame = pd.concat([j for j in data], ignore_index=True) # I add together all the datasets

# # print(len([name for name in os.listdir("Pulling_3bs") if os.path.isfile("Pulling_3bs"+'/'+name) if 'u' in name]))
# # ciao = df.median()['runs']
# # print(ciao)
# # i = np.where(df == ciao)
# # print(df.loc[df.index[df['runs'] == df['runs'].median()].tolist()])
# # print(df)
# df['runs'] = df['runs'].abs()
# # print(df)

# ciao = np.zeros(5)
# print(ciao)
# for i in range(ciao.size-1):
# 	ciao[i] = df['runs'].loc[i+1] - df['runs'].loc[i]
# print(ciao)
# print(ciao.min())
# res = np.where(ciao == ciao.min())
# print(res[0][0])
# print(df['runs'].iloc[res[0][0]])
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
# y = 1 * x_0 + 2 * x_1 + 3
y = np.array([3, 5, 7, 9, 11, 13, 15, 17]).reshape(-1, 1)
linear_regressor = LinearRegression()  # create object for the class
reg = linear_regressor.fit(X, y) 

y_pred = linear_regressor.predict(X)
print(reg.coef_[0]) # linear coefficient b
print(reg.intercept_[0]) # intercepts a 
plt.scatter(X, y, label = 'Data')
plt.plot(X, y_pred, color='red', label = 'Fit')
plt.xlabel('$\lambda$(nm)')
plt.ylabel('$f_y$(pN)')
plt.title('Linear Fit')
plt.legend()
plt.show()