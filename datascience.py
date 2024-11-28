# import numpy
# np_array = numpy.array([0,1,2,3,4,5,6,7,8,9])
# ##左闭右开原则[start,end)
# print(np_array[:5])#索引0-5不包含5 右开
# print(np_array[7:])#索引7-结尾包含7 左闭
# print(np_array[3:6])#should be [3,4,5]
#
# np_2d_array = numpy.array([[1,20],[100,5000]])
# print(np_2d_array[0,1])#2 dimension
# from array import array
#
# import numpy as np
# import numpy as np
# from timeit import Timer
#
# size_of_vec = 1000
# X_list = range(size_of_vec)
# Y_list = range(size_of_vec)
# X = np.arange(size_of_vec)
# Y = np.arange(size_of_vec)
#
# def pure_python_version():
#     Z = [X_list[i]+Y_list[i] for i in range(len(X_list))]
#
# def numpy_version():
#     Z = X+Y
# timer_obj1 = Timer("pure_python_version()",
#                    "from __main__ import pure_python_version")
# timer_obj2 = Timer("numpy_version()",
#                    "from __main__ import numpy_version")
#
# print(timer_obj1.timeit(10))
# print(timer_obj2.timeit(10))
#
# print(timer_obj1.repeat(repeat=3, number=100))
# print(timer_obj2.repeat(repeat=3,number=100))

# import pandas as pd
#
# df = pd.read_csv("/Users/wangzhengzhuo/Desktop/cellphone.csv")
#
# print(df.head())# 列名➕前5行
# #
# # print(df.columns)# 列名
# #
# # print(df.shape)
#
# print(df.iloc[0,1])
# print(df.loc[:,['Provider','Price']].head())
#
# new_row = ['C', 8, 2, 100, 100, 1]
# df.loc[8,:] = new_row
# print(df)
#
# df.drop(8, inplace=True)
#df2 = df.drop(8, inplace=False)#inplace = true直接修改df;
#inplace = False默认 先复制一个df成为新的数据库 在新数据库df2中删除第9行 而原df不变
#同时True时 只做修改 不返回任何数据 所以返回值为None
# print(df)
# # print(df2)
#
# filtered_rows = df.loc[:,'Price'] < 10
# print(filtered_rows)
#
# mean_data = df.loc[:,'Data'].mean()
# print(mean_data)
#
# grouped = df.groupby(['Provider'])
# print(grouped.size())
#
# grouped2 = df.groupby(['Provider','Voice'])#注意顺序
# print(grouped2.size())

# import numpy as np
# from numpy.ma.core import append, shape

#exercise 1
# values = []
# for i in range(0,6):
#     j = 2*i
#     values.append(j)
#
# array1 = np.array(values)
# print(array1)

#exercise 2
# array2 = np.array([[2, 1], [6, 3]])
#print(array2)

#exercise 3
# array3 = np.array([[1, 0], [0, 1]])
# multiply = np.dot(array2,array3)
# print(multiply)

#exercise 4

# import pandas as pd
#
# readin = pd.DataFrame(array2)
# print(readin)
# sum_column = readin.sum(axis= 0)
# print(sum_column)
# sum_row = readin.sum(axis= 1)
# print(sum_row)

#Chapter 10
# import pandas as pd
# Whireshark_csv = pd.read_csv("/Users/wangzhengzhuo/Desktop/")
# print(Whireshark_csv)
# csv_shape = np.shape(Whireshark_csv)
# num_column = csv_shape[1]
# print(num_column,"columns")
# num_row = csv_shape[0]
# print(num_row,"rows")
# Whireshark_csv.drop(num_row,inplace= True)
# import pandas as pd
# cellphone = pd.read_csv("/Users/wangzhengzhuo/Desktop/cellphone.csv")
# print(cellphone.columns)
# print(cellphone.shape)
# print(cellphone.iloc[1,1])#iloc index number
# print(cellphone.loc[:,'Price'])
# print(cellphone.loc[:2,:])
# new_row = ['C',8, 2, 100, 100,1]
# cellphone.loc[8,:] = new_row
# # print(cellphone)
# cellphone.drop(8, axis=0, inplace= True)
#
# cellphone.insert(6, 'NewCol',[1,2,3,4,5,6,7,8])
# new_column = cellphone.loc[:,'Price']/cellphone.loc[:,'Data']
# cellphone.drop('NewCol', axis=1, inplace=True)
# cellphone.insert(6,'GB',new_column)
# print(cellphone)
#
# filtering = cellphone.loc[:,'Price'] < 12#also .max() .mean() available
# print(filtering)
#
# grouped = cellphone.groupby(['Provider','Price'])
# print(grouped.size())
# advanced_group = cellphone.groupby(['Provider']).max()['Price']
# print(advanced_group)

#exercie for Pandas
# import pandas as pd
# Whireshark_csv = pd.read_csv("/Users/wangzhengzhuo/Desktop/capture.csv")
# print(Whireshark_csv.shape)
# num_rows = Whireshark_csv.shape[0]
# num_columns = Whireshark_csv.shape[1]
# print("Rows:", num_rows)
# print("Columns:", num_columns)
#
# print(Whireshark_csv.columns)
# # Whireshark_csv.drop('Info', axis = 1, inplace=True)
# # print(Whireshark_csv.columns)
#
# new_row = ['2', '0.000108', '10.172.9.102', '10.172.0.1', 'TCP', '66', '55564  >  53 [SYN] Seq=0 Win=64240 Len=0 MSS=1460 WS=256 SACK_PERM']
# # Whireshark_csv.loc[139,:] = new_row
# # print(Whireshark_csv.loc[139,:])
#
# new_column = 'Copy_Length'
# new_value = Whireshark_csv.loc[:,'Length']
# Whireshark_csv.insert(7, new_column, new_value)
# print(Whireshark_csv.columns)
# print(Whireshark_csv.shape)
#
# max = Whireshark_csv.loc[:,'Length'].max()
# mean = Whireshark_csv.loc[:,'Length'].mean()
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Batch size, number of classes, and epochs
batch_size = 128
num_classes = 10
epochs = 20

# Load the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data: flatten 28x28 images to 784-dimensional vectors
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Convert the data to float32 and normalize to range [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Print dataset sizes
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Model summary
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Load the dataset
columns = ["longitude", "latitude", "housingMedianAge", "totalRooms",
           "totalBedrooms", "population", "households", "medianIncome", "medianHouseValue"]
df = pd.read_csv("cal_housing.data", names=columns)

# Convert the dataset to a NumPy array
numpy_dataset = df.to_numpy()

# Split features (X) and target (y)
X = numpy_dataset[:, :8]
y = numpy_dataset[:, 8]

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Reshape y_train and y_test for compatibility
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# Standardize features and target
x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)

# Build the model
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(8,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="linear"))

# Compile the model
model.compile(loss="mse", optimizer=SGD())

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1, validation_split=0.1)

# Evaluate the model on the test set
x_test = x_scaler.transform(x_test)
y_test = y_scaler.transform(y_test)
model.evaluate(x_test, y_test)

# Predict the price for a new sample
new_sample = [[-121.10, 39.40, 21, 1200, 230, 550, 220, 5.5]]
new_sample_normalized = x_scaler.transform(new_sample)
raw_output = model.predict(new_sample_normalized)
output = y_scaler.inverse_transform(raw_output)
print(output)
