
# we will create a simple linear regression model through this code using fradient descent as optimizer and mean
#squared error as loss function

import numpy as np
import pandas as pd

# this function return the slope and the intercept of the line that fits best with your data

def gradient_desent(x,y):
    m,b = 0,0
    epochs = 10000
    lr=0.0002
    n=len(x)
    for num in range(epochs):
        y_predicted = m*x+b
        d_m = -2/n*sum(x*(y-y_predicted))
        d_b = -2/n*sum(y-y_predicted)
        m = m-lr*d_m
        b= b -lr*d_b
        cost=1/n*sum((y - y_predicted)**2)
        print('m {}, b {}, cost{}'.format(m,b,cost))
    return m,b

df = pd.read_csv(r'D:\a.csv') # your data in csv format

m,b = gradient_desent(np.array(df.math),np.array(df.cs))

print(m) #slope
print(b)# intercept

#equation of line is y = m*x+b
