import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.neural_network as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('wdbc.csv', header=None)
data = data.replace({'B':0, 'M':1})
x = data.iloc[:,2:] 
y = data.iloc[:,1]

# Scaling
x = (x - x.mean())/x.std()

# Apply PCA to reduce features from 30 to 15
pca = PCA(n_components=15)
x_pca = pca.fit_transform(x)

#Build a regression model and fit it with x, y
model = lm.LogisticRegression()
#model.fit(x_pca,y)
model.fit(x,y)

# Calculate the model score when evaluated using 5-cross-validation
#results = ms.cross_val_score(model, x_pca, y, cv=5)
results = ms.cross_val_score(model, x, y, cv=5)
print ("Regression Model Score=", results.mean())

# A sample prediction (row n)
n=101 # a sample row number
#z=pca.transform(x.iloc[n:n+1,:])
z=x.iloc[n:n+1,:]
prediction_value=model.predict(z)
prediction_class= 'M' if prediction_value.round()==1 else 'B'
actual_class= 'M' if y[n].round()==1 else 'B'
print ("The prediction for sample #" , n , " is:", prediction_value, "(" , prediction_class , ")", "Actual diagnosis was:", actual_class)

# Train a neural network model
mlp_model = nn.MLPClassifier(max_iter=700)
mlp_model.fit(x,y)

# Calculate the model score when evaluated using 5-cross-validation
results = ms.cross_val_score(mlp_model, x, y, cv=5)
print ("NN Model Score=", results.mean())

# A sample prediction (row n)
n=101 # a sample row number
z=x.iloc[n:n+1,:]
prediction_value=mlp_model.predict(z)
prediction_class= 'M' if prediction_value.round()==1 else 'B'
actual_class= 'M' if y[n].round()==1 else 'B'
print ("The prediction for sample #" , n , " is:", prediction_value, "(" , prediction_class , ")", "Actual diagnosis was:", actual_class)

weights = mlp_model.coefs_
biases = mlp_model.intercepts_

first_layer_weights = weights[0]

# print(weights)
# print the matrix sizes of the weights
print("First layer weights matrix size:", first_layer_weights.shape)
print("Second layer weights matrix size:", weights[1].shape)

plt.plot(mlp_model.loss_curve_)
plt.show()