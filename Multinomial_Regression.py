# example for multinomial regression using Convolutional Neural Network

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset=pd.read_csv('abc.csv',header=None, delim_whitespace=False)

np.random.seed(1)
X=np.asarray(dataset.iloc[:,0:3])
Y=np.asarray(dataset.iloc[:,3])

"""
scaler=StandardScaler()
scaled_X = scaler.fit_transform(X)
scaled_Y = np.rollaxis(scaler.fit_transform(Y.reshape(1,-1)),1)
#Here we are not able to use standarization of the data.
"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33,random_state=42)

model=Sequential()
model.add(Dense(4, input_dim=3))
model.add(Activation('relu'))
model.add(Dense(output_dim=1,activation='relu'))


'''
model=Sequential()
model.add(Dense(output_dim=1, input_dim=1))
model.add(Activation('relu'))
'''

model.compile(loss='mse',optimizer='adam')

print(' Training -----------')
for i in range(1001):
    cost = model.train_on_batch(X_train,Y_train)
    if i % 10 == 0:
        print('epoch',i,' train cost:', cost)


print(' Testing ------------')
cost=model.evaluate(X_test,Y_test,batch_size=40)
print(' test cost:', cost)

Y_pred = model.predict(X_test)
