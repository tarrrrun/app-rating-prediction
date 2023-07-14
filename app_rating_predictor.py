import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import pandas as pd

dff = pd.read_csv('googleplaystore.csv')
print(dff.head)

def conv_to_kb(size):
  if 'M' in size:
    return float(size.rstrip('M')) *1024
  elif 'K' in size:
    return float(size.rstrip('K'))
  else:
    return None
dff['Size']= dff['Size'].apply(conv_to_kb)

dff = dff[dff['Size'] != 'Varies with device']
dff = dff[dff['Android Ver'] != 'Varies with device']
dff = dff[dff['Current Ver'] != 'Varies with device']
dff = dff.dropna()      # Removing NaN valued rows
#print(dff['Size'])
#print(df.Rating)
#print(df)
dff['Installs']=dff['Installs'].str.replace(',','').str.replace('+','')
dff["Price"] = dff['Price'].str.replace('$','')
dff['Category'] = pd.factorize(dff['Category'])[0]          #Giving numerical representation to categorial variables by using factorize()
dff['Type'] = pd.factorize(dff['Type'])[0]
dff['Content Rating'] = pd.factorize(dff['Content Rating'])[0]
dff['Genres'] = pd.factorize(dff['Genres'])[0]
print(dff.head)

"""df_x =  df.Rating[:, np.newaxis,]
#df_x1 = df_x.dropna()
df_x_train = df_x[:-500]
df_x_test = df_x[-500:]"""
x_dff = dff[['Category' , 'Reviews', 'Size', 'Installs' , 'Type','Price', 'Content Rating', 'Genres']]
y_dff = dff['Rating']

from sklearn.model_selection import train_test_split
x_dff_train , x_dff_test, y_dff_train, y_dff_test = train_test_split(x_dff , y_dff, test_size=0.2, random_state = 42)

model = LinearRegression()                                                      # creating linear regression model
model.fit(x_dff_train,y_dff_train)                                              # Training the model
y_pre = model.predict(x_dff_test)                                               # Making predictions

from sklearn.metrics import mean_squared_error

errorr = mean_squared_error(y_dff_test,y_pre)
print('Mean squared error: ', errorr)

new_df = pd.DataFrame({'Category':[1], 'Reviews':[100],'Size':[30],'Installs':[2000],'Type':[0],'Price':['23'],'Content Rating':[10],'Genres': [0]})
pred = model.predict(new_df)
print(pred)
# Plotting scatter graph
plt.scatter(x_dff_test['Size'], y_dff_test, color='blue', label='Actual Ratings')
plt.plot(x_dff_test['Size'], y_pre, color='red', label='Predicted Ratings')
plt.xlabel('Size')
plt.ylabel('Rating')
plt.title('Actual vs Predicted Ratings')
plt.legend()
plt.show()