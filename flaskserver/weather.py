# # %%
# import numpy as np
# import pandas as pd
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.filterwarnings("ignore")
# import itertools
# import matplotlib.pyplot as plt
# import seaborn as sb
# from sklearn import metrics
# from sklearn import preprocessing
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import LinearSVC, SVC
# from sklearn.tree import DecisionTreeClassifier

# import pickle
# from datetime import datetime

# # %%
# df = pd.read_csv('./data/weather_data.csv')


# # %%
# df.head(5)

# # %%
# cols = ['temp', 'visibility', 'wind_speed']
# for column in cols:
#     df[column] = df[column].astype(float)

# def data_transform(df):
#     df = df.dropna()
#     df = df.astype({'level': 'int64'})
#     return df


# # %%
# # Transform the data
# train = data_transform(df.copy())

# # Data visualization (optional, for verification)
# plt.figure(figsize=(7, 5))
# sb.countplot(data=train, x='level')
# plt.title('Level')
# plt.show()


# # %%
# for i in cols:
#     plt.figure(figsize=(7, 5))
#     plt.hist(train[i], bins=20, color='blue', edgecolor='black')
#     plt.xlabel(f'{i}')
#     plt.title(f'{i} Histogram')
#     plt.tight_layout()
#     plt.show()

# # %%
# for i in ['visibility', 'wind_speed']:
#     sb.set(style="white") 
#     sb.jointplot(x='temp', y=i, hue='level', data=train, s=20)
#     plt.title(f'temp & {i}', pad=-10)
#     plt.show()


# # %%
# cols = ['temp', 'visibility', 'wind_speed', 'pressure', 'humidity', 'grnd_level', 'sea_level']
# sb.heatmap(train[cols].corr(), annot=True, cmap='Reds')


# # %%
# # Split the data into features and target variable
# x = train[['temp', 'visibility', 'wind_speed', 'pressure', 'humidity', 'grnd_level', 'sea_level']]
# y = train['level']

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# # %%
# # Train Linear Regression model
# md_lr = LinearRegression()
# md_lr.fit(x_train, y_train)
# pred_lr = md_lr.predict(x_test)
# discrete_pred_lr = np.round(pred_lr).astype(int)


# # %%
# # Evaluate Linear Regression model
# print(f'Linear Regression:')
# print(f'MAE : {mean_absolute_error(y_test, pred_lr)}')
# print(f'MSE : {mean_squared_error(y_test, pred_lr)}')
# print(f'RMSE : {np.sqrt(mean_squared_error(y_test, pred_lr))}')
# print(f'r2_score : {r2_score(y_test, pred_lr)}')
# print(f'Accuracy: {accuracy_score(y_test, discrete_pred_lr)}')


# # %%
# x_train

# # %%
# # Train Random Forest Regressor model
# md_rf = RandomForestRegressor(random_state=42)
# md_rf.fit(x_train, y_train)
# pred_rf = md_rf.predict(x_test)
# discrete_pred_rf = np.round(pred_rf).astype(int)

# # %%
# # Evaluate Random Forest Regressor model
# print(f'\nRandom Forest Regressor:')
# print(f'MAE : {mean_absolute_error(y_test, pred_rf)}')
# print(f'MSE : {mean_squared_error(y_test, pred_rf)}')
# print(f'RMSE : {np.sqrt(mean_squared_error(y_test, pred_rf))}')
# print(f'r2_score : {r2_score(y_test, pred_rf)}')
# print(f'Accuracy: {accuracy_score(y_test, discrete_pred_rf)}')


# # %%

# plt.rcParams["font.family"] = 'DejaVu Sans'

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=90)
#     plt.yticks(tick_marks, classes)
#     ax = plt.gca()
#     ax.set_ylim(-.5, 5.5)
        
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# # %%
# def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize=True, \
#                  print_cm=True, cm_cmap=plt.cm.Reds):
    
    
#     # to store results at various phases
#     results = dict()
    
#     # time at which model starts training 
#     train_start_time = datetime.now()
#     print('training the model..')
#     model.fit(X_train, y_train)
#     print('Done \n \n')
#     train_end_time = datetime.now()
#     results['training_time'] =  train_end_time - train_start_time
#     print('training_time(HH:MM:SS.ms) - {}\n\n'.format(results['training_time']))
    
    
#     # predict test data
#     print('Predicting test data')
#     test_start_time = datetime.now()
#     y_pred = model.predict(X_test)
#     test_end_time = datetime.now()
#     print('Done \n \n')
#     results['testing_time'] = test_end_time - test_start_time
#     print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
#     results['predicted'] = y_pred
   

#     # calculate overall accuracty of the model
#     accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
#     # store accuracy in results
#     results['accuracy'] = accuracy
#     print('---------------------')
#     print('|      Accuracy      |')
#     print('---------------------')
#     print('\n    {}\n\n'.format(accuracy))
    
#     # confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     results['confusion_matrix'] = cm
#     if print_cm: 
#         print('--------------------')
#         print('| Confusion Matrix |')
#         print('--------------------')
#         print('\n {}'.format(cm))
        
    
#     # get classification report
#     print('-------------------------')
#     print('| Classification Report |')
#     print('-------------------------')
#     classification_report = metrics.classification_report(y_test, y_pred)
#     # store report in results
#     results['classification_report'] = classification_report
#     print(classification_report)
    
#     # add the trained model to the results
#     results['model'] = model
    
#     return results


    
    

# # %%
# def print_grid_search_attributes(model):
#     # Estimator that gave highest score among all the estimators formed in GridSearch
#     print('--------------------------')
#     print('|      Best Estimator     |')
#     print('--------------------------')
#     print('\n\t{}\n'.format(model.best_estimator_))


#     # parameters that gave best results while performing grid search
#     print('--------------------------')
#     print('|     Best parameters     |')
#     print('--------------------------')
#     print('\tParameters of best estimator : \n\n\t{}\n'.format(model.best_params_))


#     #  number of cross validation splits
#     print('---------------------------------')
#     print('|   No of CrossValidation sets   |')
#     print('--------------------------------')
#     print('\n\tTotal numbre of cross validation sets: {}\n'.format(model.n_splits_))


#     # Average cross validated score of the best estimator, from the Grid Search 
#     print('--------------------------')
#     print('|        Best Score       |')
#     print('--------------------------')
#     print('\n\tAverage Cross Validate scores of best estimator : \n\n\t{}\n'.format(model.best_score_))
    

# # %% [markdown]
# # # 1. KNN with GridSearch

# # %%
# parameters = {'n_neighbors': [1, 10, 11, 20, 30]}
# log_knn = KNeighborsClassifier(n_neighbors=6)

# log_knn_grid = GridSearchCV(log_knn, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
# log_knn_grid_results =  perform_model(log_knn_grid, x_train, y_train, x_test, y_test, class_labels=cols)


# # %%
# # observe the attributes of the model 
# print_grid_search_attributes(log_knn_grid_results['model'])

# # %% [markdown]
# # #  2. Linear SVC with GridSearch

# # %%
# parameters = {'C':[0.125, 0.5, 1, 2, 8, 16]}
# lr_svc = LinearSVC(tol=0.00005)
# lr_svc_grid = GridSearchCV(lr_svc, param_grid=parameters, n_jobs=-1, verbose=1)
# lr_svc_grid_results = perform_model(lr_svc_grid, x_train, y_train, x_test, y_test, class_labels=cols)

# # %%
# print_grid_search_attributes(lr_svc_grid_results['model'])

# # %% [markdown]
# # # 3.  Kernel SVM with GridSearch

# # %%

# parameters = {'C':[2,8,16],\
#               'gamma': [ 0.0078125, 0.125, 2]}
# rbf_svm = SVC(kernel='rbf')
# rbf_svm_grid = GridSearchCV(rbf_svm,param_grid=parameters, n_jobs=-1)
# rbf_svm_grid_results = perform_model(rbf_svm_grid, x_train, y_train, x_test, y_test, class_labels=cols
#                                     )

# # %%
# print_grid_search_attributes(rbf_svm_grid_results['model'])

# # %% [markdown]
# # # 4. Decision Trees with GridSearchCV

# # %%
# parameters = {'max_depth':np.arange(3,10,2)}
# dt = DecisionTreeClassifier()
# dt_grid = GridSearchCV(dt,param_grid=parameters, n_jobs=-1)
# dt_grid_results = perform_model(dt_grid, x_train, y_train, x_test, y_test, class_labels=cols)
# print_grid_search_attributes(dt_grid_results['model'])

# # %%
# params = {'n_estimators': np.arange(10,201,20), 'max_depth':np.arange(3,15,2)}
# rfc = RandomForestClassifier()
# rfc_grid = GridSearchCV(rfc, param_grid=params, n_jobs=-1)
# rfc_grid_results = perform_model(rfc_grid, x_train, y_train, x_test, y_test, class_labels=cols)
# print_grid_search_attributes(rfc_grid_results['model'])

# # %%
# print('\n                     Accuracy     Error')
# print('                     ----------   --------')

# print('KNN                 : {:.04}%       {:.04}% '.format(log_knn_grid_results['accuracy'] * 100,\
#                                                         100-(log_knn_grid_results['accuracy'] * 100)))

# print('Linear SVC          :  {:.04}%       {:.04}% '.format(lr_svc_grid_results['accuracy'] * 100,\
#                                                         100-(lr_svc_grid_results['accuracy'] * 100)))

# print('KNN                 :  {:.04}%          {:.04}% '.format(log_knn_grid_results['accuracy'] * 100,\
#                                                         100-(log_knn_grid_results['accuracy'] * 100)))

# print('rbf SVM classifier  : {:.04}%      {:.04}% '.format(rbf_svm_grid_results['accuracy'] * 100,\
#                                                           100-(rbf_svm_grid_results['accuracy'] * 100)))

# print('DecisionTree        :  {:.04}%      {:.04}% '.format(dt_grid_results['accuracy'] * 100,\
#                                                         100-(dt_grid_results['accuracy'] * 100)))

# print('Random Forest       : {:.04}%      {:.04}% '.format(rfc_grid_results['accuracy'] * 100,\
#                                                            100-(rfc_grid_results['accuracy'] * 100)))

# # %%


# # %%
# import pickle
# from sklearn.ensemble import RandomForestClassifier

# # Retrain the model with the best parameters
# best_max_depth = 13
# best_n_estimators = 50

# rfc_best = RandomForestClassifier(max_depth=best_max_depth, n_estimators=best_n_estimators)
# rfc_best.fit(x_train, y_train)

# # Export the trained model using pickle
# with open('./trained models/weather/rfc_model.pkl', 'wb') as file:
#     pickle.dump(rfc_best, file)


# # %%
# x_train

# # %%
# import pickle
# import numpy as np

# # Load the trained model
# try:
#     with open('./rfc_model.pkl', 'rb') as file:
#         model = pickle.load(file)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None




# # %%














# %%
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import itertools
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

import pickle
from datetime import datetime

# %%
df = pd.read_csv('./data/weather_data_.csv')


# %%
df.head(5)

# %%
cols = ['temp', 'visibility', 'wind_speed']
for column in cols:
    df[column] = df[column].astype(float)

def data_transform(df):
    df = df.dropna()
    df = df.astype({'level': 'int64'})
    return df


# %%
# Transform the data
train = data_transform(df.copy())

# Data visualization (optional, for verification)
plt.figure(figsize=(7, 5))
sb.countplot(data=train, x='level')
plt.title('Level')
plt.show()


# %%
for i in cols:
    plt.figure(figsize=(7, 5))
    plt.hist(train[i], bins=20, color='blue', edgecolor='black')
    plt.xlabel(f'{i}')
    plt.title(f'{i} Histogram')
    plt.tight_layout()
    plt.show()

# %%
for i in ['visibility', 'wind_speed']:
    sb.set(style="white") 
    sb.jointplot(x='temp', y=i, hue='level', data=train, s=20)
    plt.title(f'temp & {i}', pad=-10)
    plt.show()


# %%
cols = ['temp', 'visibility', 'wind_speed', 'pressure', 'humidity', 'grnd_level', 'sea_level']
sb.heatmap(train[cols].corr(), annot=True, cmap='Reds')


# %%
# Split the data into features and target variable
x = train[['temp', 'visibility', 'wind_speed', 'pressure', 'humidity', 'grnd_level', 'sea_level']]
y = train['level']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# %%
# Train Linear Regression model
md_lr = LinearRegression()
md_lr.fit(x_train, y_train)
pred_lr = md_lr.predict(x_test)
discrete_pred_lr = np.round(pred_lr).astype(int)


# %%
# Evaluate Linear Regression model
print(f'Linear Regression:')
print(f'MAE : {mean_absolute_error(y_test, pred_lr)}')
print(f'MSE : {mean_squared_error(y_test, pred_lr)}')
print(f'RMSE : {np.sqrt(mean_squared_error(y_test, pred_lr))}')
print(f'r2_score : {r2_score(y_test, pred_lr)}')
print(f'Accuracy: {accuracy_score(y_test, discrete_pred_lr)}')


# %%
x_train

# %%
# Train Random Forest Regressor model
md_rf = RandomForestRegressor(random_state=42)
md_rf.fit(x_train, y_train)
pred_rf = md_rf.predict(x_test)
discrete_pred_rf = np.round(pred_rf).astype(int)

# %%
# Evaluate Random Forest Regressor model
print(f'\nRandom Forest Regressor:')
print(f'MAE : {mean_absolute_error(y_test, pred_rf)}')
print(f'MSE : {mean_squared_error(y_test, pred_rf)}')
print(f'RMSE : {np.sqrt(mean_squared_error(y_test, pred_rf))}')
print(f'r2_score : {r2_score(y_test, pred_rf)}')
print(f'Accuracy: {accuracy_score(y_test, discrete_pred_rf)}')


# %%

plt.rcParams["font.family"] = 'DejaVu Sans'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    ax = plt.gca()
    ax.set_ylim(-.5, 5.5)
        
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# %%
def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize=True, \
                 print_cm=True, cm_cmap=plt.cm.Reds):
    
    
    # to store results at various phases
    results = dict()
    
    # time at which model starts training 
    train_start_time = datetime.now()
    print('training the model..')
    model.fit(X_train, y_train)
    print('Done \n \n')
    train_end_time = datetime.now()
    results['training_time'] =  train_end_time - train_start_time
    print('training_time(HH:MM:SS.ms) - {}\n\n'.format(results['training_time']))
    
    
    # predict test data
    print('Predicting test data')
    test_start_time = datetime.now()
    y_pred = model.predict(X_test)
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted'] = y_pred
   

    # calculate overall accuracty of the model
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # store accuracy in results
    results['accuracy'] = accuracy
    print('---------------------')
    print('|      Accuracy      |')
    print('---------------------')
    print('\n    {}\n\n'.format(accuracy))
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm: 
        print('--------------------')
        print('| Confusion Matrix |')
        print('--------------------')
        print('\n {}'.format(cm))
        
    
    # get classification report
    print('-------------------------')
    print('| Classification Report |')
    print('-------------------------')
    classification_report = metrics.classification_report(y_test, y_pred)
    # store report in results
    results['classification_report'] = classification_report
    print(classification_report)
    
    # add the trained model to the results
    results['model'] = model
    
    return results


    
    

# %%
def print_grid_search_attributes(model):
    # Estimator that gave highest score among all the estimators formed in GridSearch
    print('--------------------------')
    print('|      Best Estimator     |')
    print('--------------------------')
    print('\n\t{}\n'.format(model.best_estimator_))


    # parameters that gave best results while performing grid search
    print('--------------------------')
    print('|     Best parameters     |')
    print('--------------------------')
    print('\tParameters of best estimator : \n\n\t{}\n'.format(model.best_params_))


    #  number of cross validation splits
    print('---------------------------------')
    print('|   No of CrossValidation sets   |')
    print('--------------------------------')
    print('\n\tTotal numbre of cross validation sets: {}\n'.format(model.n_splits_))


    # Average cross validated score of the best estimator, from the Grid Search 
    print('--------------------------')
    print('|        Best Score       |')
    print('--------------------------')
    print('\n\tAverage Cross Validate scores of best estimator : \n\n\t{}\n'.format(model.best_score_))
    

# %% [markdown]
# # 1. KNN with GridSearch

# %%
parameters = {'n_neighbors': [1, 10, 11, 20, 30]}
log_knn = KNeighborsClassifier(n_neighbors=6)

log_knn_grid = GridSearchCV(log_knn, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
log_knn_grid_results =  perform_model(log_knn_grid, x_train, y_train, x_test, y_test, class_labels=cols)


# %%
# observe the attributes of the model 
print_grid_search_attributes(log_knn_grid_results['model'])

# %% [markdown]
# #  2. Linear SVC with GridSearch

# %%
parameters = {'C':[0.125, 0.5, 1, 2, 8, 16]}
lr_svc = LinearSVC(tol=0.00005)
lr_svc_grid = GridSearchCV(lr_svc, param_grid=parameters, n_jobs=-1, verbose=1)
lr_svc_grid_results = perform_model(lr_svc_grid, x_train, y_train, x_test, y_test, class_labels=cols)

# %%
print_grid_search_attributes(lr_svc_grid_results['model'])

# %% [markdown]
# # 3.  Kernel SVM with GridSearch

# %%

parameters = {'C':[2,8,16],\
              'gamma': [ 0.0078125, 0.125, 2]}
rbf_svm = SVC(kernel='rbf')
rbf_svm_grid = GridSearchCV(rbf_svm,param_grid=parameters, n_jobs=-1)
rbf_svm_grid_results = perform_model(rbf_svm_grid, x_train, y_train, x_test, y_test, class_labels=cols
                                    )

# %%
print_grid_search_attributes(rbf_svm_grid_results['model'])

# %% [markdown]
# # 4. Decision Trees with GridSearchCV

# %%
parameters = {'max_depth':np.arange(3,10,2)}
dt = DecisionTreeClassifier()
dt_grid = GridSearchCV(dt,param_grid=parameters, n_jobs=-1)
dt_grid_results = perform_model(dt_grid, x_train, y_train, x_test, y_test, class_labels=cols)
print_grid_search_attributes(dt_grid_results['model'])

# %%
params = {'n_estimators': np.arange(10,201,20), 'max_depth':np.arange(3,15,2)}
rfc = RandomForestClassifier()
rfc_grid = GridSearchCV(rfc, param_grid=params, n_jobs=-1)
rfc_grid_results = perform_model(rfc_grid, x_train, y_train, x_test, y_test, class_labels=cols)
print_grid_search_attributes(rfc_grid_results['model'])

# %%
print('\n                     Accuracy     Error')
print('                     ----------   --------')

print('KNN                 : {:.04}%       {:.04}% '.format(log_knn_grid_results['accuracy'] * 100,\
                                                        100-(log_knn_grid_results['accuracy'] * 100)))

print('Linear SVC          :  {:.04}%       {:.04}% '.format(lr_svc_grid_results['accuracy'] * 100,\
                                                        100-(lr_svc_grid_results['accuracy'] * 100)))

print('KNN                 :  {:.04}%          {:.04}% '.format(log_knn_grid_results['accuracy'] * 100,\
                                                        100-(log_knn_grid_results['accuracy'] * 100)))

print('rbf SVM classifier  : {:.04}%      {:.04}% '.format(rbf_svm_grid_results['accuracy'] * 100,\
                                                          100-(rbf_svm_grid_results['accuracy'] * 100)))

print('DecisionTree        :  {:.04}%      {:.04}% '.format(dt_grid_results['accuracy'] * 100,\
                                                        100-(dt_grid_results['accuracy'] * 100)))

print('Random Forest       : {:.04}%      {:.04}% '.format(rfc_grid_results['accuracy'] * 100,\
                                                           100-(rfc_grid_results['accuracy'] * 100)))

# %%


# %%
import pickle
from sklearn.ensemble import RandomForestClassifier

# Retrain the model with the best parameters
best_max_depth = 13
best_n_estimators = 50

rfc_best = RandomForestClassifier(max_depth=best_max_depth, n_estimators=best_n_estimators)
rfc_best.fit(x_train, y_train)

# Export the trained model using pickle
with open('./trained models/weather/rfc_model.pkl', 'wb') as file:
    pickle.dump(rfc_best, file)


# %%
x_train

# %%
import pickle
import numpy as np

# Load the trained model
try:
    with open('./rfc_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Example input features
input_features = [305.50, 3081.0, 2.87, 985, 28, 966, 1014]

# Convert input features to numpy array
features = np.array(input_features).reshape(1, -1)

# Make prediction
prediction = model.predict(features)[0]

# Print prediction
print(int(prediction))


# %%



