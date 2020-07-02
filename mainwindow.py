import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df =pd.read_csv(r"D:\training basket\Datasets\homeprices.csv")
df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
data_x = df.iloc[:,0:-1].values

data_y = df.iloc[:,-1].values

#print(data_y)

X_train,X_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0)

reg = LinearRegression()
reg.fit(X_train,y_train)

print("Train Score:", reg.score(X_train,y_train))
print("Test Score:", reg.score(X_test,y_test))

pickle.dump(reg, open('home.pkl','wb'))

model = pickle.load(open('home.pkl','rb'))
print(model.predict([[1200,8,10]]))
