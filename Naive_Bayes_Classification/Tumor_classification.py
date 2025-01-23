import pandas as pd
import numpy as np
import scipy.stats as s
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv("data.csv")

data.head()
data.info()

data.drop(labels=data.columns[32],axis=1,inplace=True)
data_columns = data.columns
data_columns = list(data_columns)

data_columns.remove('id')
data_columns.remove('diagnosis')

print(data_columns)

def plot_grid_histplot(data,data_columns,shape,figure_size):

    data_columns = np.array(data_columns).reshape(shape[0],shape[1])
    fig , axes = plt.subplots(shape[0],shape[1],figsize=figure_size)

    for i in range (data_columns.shape[0]):
        for j in range (data_columns.shape[1]):
            sns.histplot(data=data,x=data_columns[i,j],hue='diagnosis',stat='density',bins=10,kde=True,
                         palette=[sns.color_palette()[3],sns.color_palette()[0]],element='step',ax=axes[i,j])
            
def plot_grid_histplot(data,data_columns,shape,figure_size):

    data_columns = np.array(data_columns).reshape(shape[0],shape[1])
    fig , axes = plt.subplots(shape[0],shape[1],figsize=figure_size)

    for i in range (data_columns.shape[0]):
        for j in range (data_columns.shape[1]):
            sns.histplot(data=data,x=data_columns[i,j],hue='diagnosis',stat='density',bins=10,kde=True,
                         palette=[sns.color_palette()[3],sns.color_palette()[0]],element='step',ax=axes[i,j])
            
data_copy = data.replace(to_replace=['B','M'] , value=[0,1] , inplace=False)
data_corr = data_copy.corr()
data_corr

mask = np.zeros_like(data_copy.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f , ax = plt.subplots(figsize=(35,25))
    sns.heatmap(data=data_corr,vmin=0,vmax=1,mask=mask,square=True,annot=True)

strong_relation_features = pd.Series(data_corr['diagnosis']).nlargest(n=9).iloc[1:]
strong_relation_features

diagnosis = data_copy['diagnosis']
data_copy = data_copy[list(strong_relation_features.to_dict().keys())]

data_copy['diagnosis'] = diagnosis
data_copy

mask = np.zeros_like(data_copy.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f , ax = plt.subplots(figsize=(7,7))
    sns.heatmap(data=data_copy.corr(),vmin=0,vmax=1,mask=mask,square=True,annot=True)

data_copy_cov = np.array(data_copy[list(strong_relation_features.to_dict().keys())].cov())
data_copy_cov

data_copy_cov_det = np.linalg.det(data_copy_cov)
data_copy_cov_det
data_copy_cov.shape

#number of benaine tumers
data_copy[data_copy['diagnosis']==0].shape[0]

#number of maglinent tumers
data_copy[data_copy['diagnosis']==1].shape[0]

#split the data into traning and crossvalidation data set
class0_data = data_copy[data_copy['diagnosis']==0]
class1_data = data_copy[data_copy['diagnosis']==1]

class0_traning_data = class0_data.iloc[0:int(0.75*len(class0_data))]
class1_traning_data = class1_data.iloc[0:int(0.75*len(class1_data))]

class0_cv_data = class0_data.iloc[int(0.75*len(class0_data)):]
class1_cv_data = class1_data.iloc[int(0.75*len(class1_data)):]

traning_data = pd.concat([class0_traning_data , class1_traning_data])
cv_data = pd.concat([class0_cv_data , class1_cv_data])

#Maximum liklihood estimates for mu0 and sigma0
mu_0 = np.array(traning_data[traning_data['diagnosis']==0].iloc[:,0:8].mean())
sigma_0 = np.array(traning_data[traning_data['diagnosis']==0].iloc[:,0:8].cov())

print(mu_0)
print("\n")
print(sigma_0)

#maximum liklihood estimate for mu1 and sigma1
mu_1 = np.array(traning_data[traning_data['diagnosis']==1].iloc[:,0:8].mean())
sigma_1 = np.array(traning_data[traning_data['diagnosis']==1].iloc[:,0:8].cov())

print(mu_1)
print("\n")
print(sigma_1)

def predict_classes(data):

    prob_xi_given_class1 = s.multivariate_normal.pdf(data,mu_1,sigma_1)

    prob_xi_given_class0 = s.multivariate_normal.pdf(data,mu_0,sigma_0)

    prob_class1_given_xi = prob_xi_given_class1/(prob_xi_given_class1 + prob_xi_given_class0)

    return prob_class1_given_xi > 0.5

predicted_classes = predict_classes(cv_data.iloc[:,0:8])

predicted_classes

confusion_matrix(y_true=cv_data['diagnosis'],y_pred=predicted_classes)
print(classification_report(y_true=cv_data['diagnosis'],y_pred=predicted_classes))