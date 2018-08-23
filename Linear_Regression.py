# Implementing the regression problem with the help of kaggle.com

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# generate 2d classification dataset

X = np.linspace(-1, 1, 200) 
np.random.shuffle(X)
Y=0.5 * X + 2 + np.random.normal(0, 0.05, (200,))
#plt.scatter(X,Y)
#plt.show()
#sns.barplot(X,Y,palette=sns.cubehelix_palette(len(X)))

#splitting the dataset into training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33,random_state=42)

model=Sequential()
model.add(Dense(output_dim=1, input_dim=1))
model.compile(loss='mse',optimizer='sgd')

print(' Training -----------')
for i in range(1001):
    cost = model.train_on_batch(X_train,Y_train)
    if i % 100 == 0:
        print(' train cost:', cost)


print(' Testing ------------')
cost=model.evaluate(X_test,Y_test,batch_size=40)
print(' test cost:', cost)
W,b = model.layers[0].get_weights()
print(' Weights=', W, '\nbiases=', b)

# The output of the simple linear regression is in the form of f(X)= W  * X + b

Y_pred = model.predict(X_test)
plt.scatter(X_train,Y_train)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
