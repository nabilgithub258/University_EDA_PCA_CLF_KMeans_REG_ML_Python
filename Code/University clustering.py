#!/usr/bin/env python
# coding: utf-8

# In[591]:


#####################################################################################################
######################### UNIVERSITY DATA SET  ######################################################
#####################################################################################################


# In[592]:


#################################################################
############ Part I - Importing
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[593]:


df = pd.read_csv('College_Data',index_col=0)


# In[594]:


#####################################################################
########################### Part II - Duplicates
#####################################################################


# In[595]:


df[df.duplicated()]


# In[596]:


df.head()


# In[597]:


####################################################################
############## Part III - Missing Values
####################################################################


# In[598]:


from matplotlib.colors import LinearSegmentedColormap

Amelia = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])


# In[599]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### why Amelia, if you coming from R then you might have used Amelia package which detects the missing value 
#### On July 2, 1937, Amelia disappeared over the Pacific Ocean while attempting to become the first female pilot to circumnavigate the world


# In[600]:


df.isnull().any()                 #### no missing values found


# In[601]:


df.info()


# In[602]:


df.head()


# In[603]:


######################################################################
############## Part IV - EDA
######################################################################


# In[604]:


df['Accept'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='green',color='black')
df['Apps'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('University Accept Graph')

plt.xlabel('University')

plt.ylabel('Density')

#### seems like one university is getting a lot of application and hence the acceptence is higher as well


# In[605]:


df[df.Accept == df.Accept.max()]                  #### its Rutgers, being from NJ its quite suprising to me honestly


# In[606]:


df['Enroll'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',color='black')

plt.title('University Enroll Graph')

plt.xlabel('University')

plt.ylabel('Density')

#### this is more realistic graph because it details the numbers of applicants enrolled


# In[607]:


df[df.Enroll == df.Enroll.min()]                #### this is the least enrolled university/college in our data set.\
                                                #### also the application they getting is just 100 out of which they only reject 10
                                                #### its private so that should contribute more to its lesser enrollment
    


# In[608]:


df[df.Enroll == df.Enroll.max()]                    #### interesting but its not private so that should be the reason but still interesting


# In[609]:


df.plot(x='Grad.Rate',y=['Apps','Accept','Enroll'],linestyle='',marker='o',figsize=(20,7),color={'Apps':'red',
                                                                                                 'Accept':'black',
                                                                                                 'Enroll':'green'})

#### so obviously the applications are massive but graduation rate is lower when the applicants are less which is not suprising
#### but there is some mistake in this data set, graduation rate can't exceed 100% so lets take that into account


# In[610]:


df = df[df['Grad.Rate'] <= 100]

df


# In[611]:


df['Grad.Rate'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='green',color='black')

plt.title('University Graduation Rate Graph')

plt.xlabel('University')

plt.ylabel('Density')

#### lets see who are below 20% grad rate


# In[612]:


df[df['Grad.Rate'] < 20]          #### all private, interesting


# In[613]:


df['private_uni'] = df.Private.map({'Yes':'1',
                                    'No':0})

df.private_uni.unique()


# In[614]:


df.private_uni.value_counts()


# In[615]:


df.info()


# In[616]:


df['private_uni'] = df['private_uni'].astype(int)


# In[617]:


g = sns.jointplot(x='Grad.Rate',y='private_uni',data=df,kind='reg',x_bins=[1,5,10,15,18,20,25,30,40,45,60,70,80,90,100],color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)

g.ax_joint.set_ylim(0,1)

#### clearly we see some correlation here, as the increment to Private university increases the graduation rate increases too


# In[618]:


from scipy.stats import pearsonr


# In[619]:


co_eff, p_value = pearsonr(df['Grad.Rate'],df.private_uni)

co_eff


# In[620]:


p_value                          #### definately correlated 


# In[621]:


pl = sns.FacetGrid(df,hue='Private',aspect=4,height=4,palette='winter')

pl.map(sns.kdeplot,'Outstate',fill=True)

pl.set(xlim=(0,df.Outstate.max()))

pl.add_legend()

#### this is not suprising at all, if you are outstate student in a Private university, you end up paying susbtantially more then if you were outstate in a non private university


# In[622]:


custom = {'Yes':'purple',
          'No':'green'}

pl = sns.FacetGrid(df,hue='Private',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'Top10perc',fill=True)

pl.set(xlim=(0,df.Top10perc.max()))

pl.add_legend()

#### interesting students who are top10 percent prefer private universities, in short students who are excellent in their academics prefer private universities compared to non privates


# In[623]:


corr = df.corr()

corr


# In[624]:


fig, ax = plt.subplots(figsize=(30,14))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')

#### the ones to watch are Outstate, top10perc, top25perc, roomboard, perc_alumni, expend, grad_rate


# In[625]:


sns.lmplot(x='Outstate',y='private_uni',data=df,height=7,aspect=2,x_bins=[2340,3000,4000,5000,7000,9000,11000,14000,17000,19000,20000,21000,21700],line_kws={'color':'green'},scatter_kws={'color':'black'})

#### this is not suprising at all, the Outstate fees increases as the y-axis appraches private universities


# In[626]:


sns.catplot(x='Private',y='Top25perc',data=df,kind='strip',height=7,aspect=2,legend=True,jitter=True,color='black')

#### seems like best students prefer the private universities more compared to non private ones, very interesting


# In[627]:


custom = {'Yes':'orange',
          'No':'red'}

pl = sns.FacetGrid(df,hue='Private',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'Room.Board',fill=True)

pl.set(xlim=(0,df['Room.Board'].max()))

pl.add_legend()

#### obviosuly if the university is private the fees for room boarding is 


# In[628]:


sns.catplot(x='Private',data=df,kind='count',height=7,aspect=2,palette={'Yes':'purple',
                                                                        'No':'pink'})

#### interestingly we have more students in private universities then in non privates


# In[629]:


sns.catplot(x='Private',y='perc.alumni',data=df,kind='box',height=7,aspect=2,legend=True,palette={'Yes':'red',
                                                                                                  'No':'green'})

#### another interesting aspect for private universities is that alumni do contribute significantly more towards private universities


# In[630]:


sns.lmplot(x='perc.alumni',y='Outstate',data=df,hue='Private',height=7,aspect=2,palette={'Yes':'black',
                                                                                         'No':'red'})

#### we see a clear correlation between Private uni to alumni donation and outstate


# In[631]:


sns.catplot(x='perc.alumni',y='Grad.Rate',data=df,kind='box',height=7,aspect=2,legend=True,hue='Private',palette='Set2')

#### we clearly see the graduation rate favors private universities but also donation from alumni is significantly higher from private universities alumni


# In[632]:


custom = {'Yes':'purple',
          'No':'pink'}

g = sns.jointplot(x='perc.alumni',y='Top10perc',data=df,hue='Private',kind='kde',fill=True,palette=custom)

g.fig.set_size_inches(17,9)

#### here again we see the density for top10 students enrolled in Private is more


# In[633]:


sns.catplot(x='Grad.Rate',y='Outstate',data=df,kind='box',height=7,aspect=2,legend=True,hue='Private',palette=custom)

#### clearly we see that grad rate and outstate is overall higher in private universities


# In[634]:


custom = {0:'purple',
         1:'red'}

g = sns.jointplot(x=df.Top10perc,y=df.Top25perc,data=df,hue='private_uni',palette=custom)

g.fig.set_size_inches(17,9)

#### definately we see more ratio to both variables in terms of private univeristies


# In[635]:


heat = df.groupby(['Top10perc','Grad.Rate'])['private_uni'].sum().unstack().fillna(0)

heat


# In[636]:


fig, ax = plt.subplots(figsize=(90,42))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis')

#### a heatmap for private universities with regards to top10 students enrolled and grduation rate


# In[637]:


df.groupby(['Grad.Rate'])['private_uni'].sum().fillna(0).plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',markersize=15,linestyle='dashed',linewidth=4,color='red')

#### seems like the highest number of private uni we see is above 20 and grad rate between 70-80


# In[638]:


######################################################################
############## Part V - PCA
######################################################################


# In[639]:


df.reset_index(inplace=True)


# In[640]:


df.head()


# In[641]:


df.drop(columns='index',inplace=True)


# In[642]:


df.head()


# In[643]:


X = df.drop(columns=['Private','private_uni'])

X.head()


# In[644]:


y = df['private_uni']

y.head()


# In[645]:


from sklearn.preprocessing import StandardScaler


# In[646]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[647]:


from sklearn.decomposition import PCA


# In[648]:


pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])
final_df = pd.concat([principal_df, y], axis=1)

final_df.head()


# In[649]:


final_df.isnull().any()


# In[650]:


colors = {0: 'green', 1: 'black'}

plt.figure(figsize=(15, 6))

for i in final_df['private_uni'].unique():
    subset = final_df[final_df['private_uni'] == i]
    plt.scatter(subset['principal_component_1'], subset['principal_component_2'], 
                color=colors[i], label=f'private_uni = {i}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Univesity Data set')
plt.legend()
plt.grid(True)

#### beauty of PCA


# In[651]:


features = X.columns

features


# In[652]:


df_comp = pd.DataFrame(pca.components_,columns=[features])

df_comp


# In[653]:


fig, ax = plt.subplots(figsize=(20,8))                     

sns.heatmap(df_comp,ax=ax,linewidths=0.5,annot=True,cmap='viridis')

#### PCA corr heatmap


# In[654]:


#######################################################################
######################## Part VI - Model - Classification
#######################################################################


# In[655]:


df.head()


# In[656]:


df.drop(columns='Private',inplace=True)                   #### we need to drop categorical cols for vif


# In[657]:


df.head()


# In[658]:


from statsmodels.tools.tools import add_constant

df_with_constant = add_constant(df)

df_with_constant.head()                    #### setting up Vif


# In[659]:


vif = pd.DataFrame() 


# In[660]:


vif["Feature"] = df_with_constant.columns


# In[661]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif["VIF"] = [variance_inflation_factor(df_with_constant.values, i) for i in range(df_with_constant.shape[1])]


# In[662]:


vif                #### this is bad news, but we wouldnt be dropping anything but instead take care of it using pipeline


# In[663]:


from sklearn.model_selection import train_test_split


# In[664]:


X = df.drop(columns=['private_uni'])

X.head()


# In[665]:


y = df['private_uni']

y.value_counts()                      #### not the best data, small data and to top it off its imbalanced target


# In[666]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# In[667]:


X.columns


# In[668]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components='mle'))  
        ]),['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad',
       'P.Undergrad', 'Outstate', 'Room.Board', 'Books', 'Personal', 'PhD',
       'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'Grad.Rate'])
    ])


# In[669]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[670]:


from sklearn.linear_model import LogisticRegression


# In[671]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[672]:


model.fit(X_train,y_train)


# In[673]:


y_predict = model.predict(X_test)


# In[674]:


from sklearn import metrics


# In[675]:


print(metrics.classification_report(y_test,y_predict))                #### quite decent model for the first try


# In[676]:


from sklearn.ensemble import RandomForestClassifier                #### bringing random forest


# In[677]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])


# In[678]:


model.fit(X_train,y_train)


# In[679]:


y_predict = model.predict(X_test)


# In[680]:


print(metrics.classification_report(y_test,y_predict))                #### made it worst


# In[681]:


from sklearn.linear_model import RidgeClassifier                     #### lets see what ridge can bring to the table


# In[682]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad',
       'P.Undergrad', 'Outstate', 'Room.Board', 'Books', 'Personal', 'PhD',
       'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'Grad.Rate'])
    ])


# In[683]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RidgeClassifier(alpha=1.0))
])


# In[684]:


model.fit(X_train,y_train)


# In[685]:


y_predict = model.predict(X_test)


# In[686]:


print(metrics.classification_report(y_test,y_predict))                #### quite decent


# In[687]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components='mle'))  
        ]),['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad',
       'P.Undergrad', 'Outstate', 'Room.Board', 'Books', 'Personal', 'PhD',
       'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'Grad.Rate'])
    ])


# In[688]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[689]:


import xgboost as xgb


# In[690]:


clf_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',n_jobs=-1))
])

param_grid_xgb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.7, 0.8, 0.9],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9]
}


# In[691]:


from sklearn.model_selection import RandomizedSearchCV


# In[692]:


get_ipython().run_cell_magic('time', '', "\nrandom_search_xgb = RandomizedSearchCV(clf_xgb, param_grid_xgb, cv=3, scoring='accuracy', random_state=42,verbose=2)\nrandom_search_xgb.fit(X_train, y_train)")


# In[693]:


best_model = random_search_xgb.best_estimator_


# In[694]:


y_predict = best_model.predict(X_test)


# In[695]:


print(metrics.classification_report(y_test,y_predict))                           #### best one yet


# In[696]:


from imblearn.over_sampling import SMOTE


# In[697]:


from imblearn.pipeline import Pipeline as ImbPipeline


# In[698]:


model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1))
])


# In[699]:


model.fit(X_train, y_train)


# In[700]:


y_predict = model.predict(X_test)


# In[701]:


cm = metrics.confusion_matrix(y_test,y_predict)

labels = ['Public','Private']

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

fig, ax = plt.subplots(figsize=(25,12))

disp.plot(ax=ax)


# In[702]:


from lightgbm import LGBMClassifier


# In[703]:


model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', LGBMClassifier(class_weight='balanced', random_state=42,n_jobs=-1))
])


# In[704]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[705]:


y_predict = model.predict(X_test)


# In[706]:


print(metrics.classification_report(y_test,y_predict))                  #### its not improving much anymore 


# In[707]:


#### now lets suppose we were presented this data set without target columns and we need to figure out a way to cluster them into groups of 2 as private and public
#### therefore we will be using KMeans


# In[708]:


from sklearn.cluster import KMeans


# In[709]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KMeans(n_clusters=2,random_state=42))
])


# In[710]:


model.fit(X_train,y_train)


# In[711]:


y_predict = model.predict(X_test)


# In[712]:


print(metrics.classification_report(y_test,y_predict))                 #### we have to remember that KMeans is an unsupervised classifier so its only used when the target column is not known or defined
                                                                       #### using here doesn't make sense also its not used to compare to target columns like we did here


# In[713]:


#######################################################################
######################## Part VI - Model - Regression
#######################################################################


# In[714]:


df.head()


# In[715]:


X = df.drop(columns='Grad.Rate')

X.head()                        #### we will be predicting the graduation rate here


# In[716]:


y = df['Grad.Rate']

y.head()


# In[717]:


from sklearn.linear_model import LinearRegression


# In[718]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components='mle'))  
        ]),['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad',
       'P.Undergrad', 'Outstate', 'Room.Board', 'Books', 'Personal', 'PhD',
       'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'private_uni'])
    ])


# In[719]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[720]:


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression(n_jobs=-1))
                       ])


# In[721]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[722]:


y_predict = model.predict(X_test)


# In[723]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')

#### looks good


# In[724]:


metrics.r2_score(y_test,y_predict)


# In[725]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))              #### we are off by 12 in graduation rate which honestly is OK


# In[726]:


from sklearn.ensemble import RandomForestRegressor


# In[727]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42,max_features='auto',n_estimators=100,n_jobs=-1))
])


# In[728]:


param_grid = {
    'regressor__n_estimators': [50,100],
    'regressor__max_depth': [None, 10],
    'regressor__min_samples_split': [2],
    'regressor__min_samples_leaf': [1]
}


# In[729]:


from sklearn.model_selection import GridSearchCV


# In[730]:


grid_model = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=2)


# In[731]:


get_ipython().run_cell_magic('time', '', '\ngrid_model.fit(X_train, y_train)')


# In[732]:


best_model = grid_model.best_estimator_


# In[733]:


y_predict = best_model.predict(X_test)


# In[734]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')


# In[735]:


metrics.r2_score(y_test,y_predict)


# In[736]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))              #### we are off by 13 so not better then before


# In[737]:


from sklearn.model_selection import RandomizedSearchCV


# In[738]:


param_grid = {
    'regressor__n_estimators': [100, 200, 500],
    'regressor__max_features': ['auto', 'sqrt'],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
    'regressor__bootstrap': [True, False]
}


# In[739]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42,max_features='auto',n_estimators=100,n_jobs=-1))
])


# In[740]:


random_search = RandomizedSearchCV(model, param_grid, cv=3, random_state=42, verbose=2)


# In[741]:


get_ipython().run_cell_magic('time', '', '\nrandom_search.fit(X_train, y_train)')


# In[742]:


best_model = random_search.best_estimator_


# In[743]:


y_predict = best_model.predict(X_test)


# In[744]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')


# In[745]:


metrics.r2_score(y_test,y_predict)


# In[746]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))              #### no improvement


# In[747]:


df.head()


# In[748]:


X = df.drop(columns=['Grad.Rate','private_uni'])

X.head()                        #### we are preventing data leakage by not showing private_uni in this case


# In[749]:


y = df['Grad.Rate']

y.head()                      


# In[750]:


from sklearn.linear_model import LinearRegression


# In[751]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components='mle'))  
        ]),['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad',
       'P.Undergrad', 'Outstate', 'Room.Board', 'Books', 'Personal', 'PhD',
       'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend'])
    ])


# In[752]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[753]:


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression(n_jobs=-1))
                       ])


# In[754]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[755]:


y_predict = model.predict(X_test)


# In[756]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')


# In[757]:


metrics.r2_score(y_test,y_predict)


# In[758]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))              #### this is the best one yet


# In[ ]:


############################################################################################################################
#### We successfully concluded our university dataset project, achieving strong classification results with the ############
#### GradientBoostingClassifier, which delivered 91% accuracy and precision. Despite the small and imbalanced dataset,######
#### our approach with SMOTE also performed well. In regression, we obtained an RMSE of 12 using logistic regression,#######
#### carefully excluding the private_uni variable to prevent data leakage. Overall, the project yielded valuable insights,##
#### with solid outcomes in both classification and regression tasks. ######################################################
############################################################################################################################

