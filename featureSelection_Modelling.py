import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.feature_selection import  SelectKBest, chi2
import itertools
from sklearn.linear_model import Lasso
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import cross_validation

# from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

# Filter the features that are correlated.
def remove_correlated_features():
    features = df.copy().columns.tolist()
    # Removing the features that are not related to stars/rating
    non_relevant_attributes = ["business_id","address","city","state","postal_code","name","type","stars"]
    for item in non_relevant_attributes:
        features.remove(item)
    
    # Checking for features that are correlated(correlation greater than 70%)
    correlated_attribute = df.corr(method='pearson')
    indices = np.where(abs(correlated_attribute) > 0.7)
    indices = [(correlated_attribute.index[x], correlated_attribute.columns[y]) for x, y in zip(*indices)
                                        if x != y and x < y]
    print 'Correlated features are::',indices
    # Based on above observation, Categories_Coffee&Tea & Categories_Italian features are correlated. Hence we consider one of these  
    # i.e Categories_Italian & ignore Categories_Coffee&Tea since it is similar to Categories_Italian
    features.remove("Categories_Coffee&Tea")
    features.insert(0,"stars")
    return features
   
# Finding top 10 features.
def top_features(feature_set):

    print '\nSelecting top 10 features from ::',feature_set.columns.tolist()
    
    # 1. Removing features with low variance.
    remaining_columns = feature_set.columns
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit(reduced_df)
    feature_indices = sel.get_support(indices=True)
    feature_names = [remaining_columns[idx] for idx, _  in enumerate(remaining_columns) if idx in feature_indices]
    print '\nFeature set after removal of features having low variance :: ', feature_names
    redundant_features = list(set(feature_set.columns.tolist()) - set(feature_names))
    feature_set = feature_set.drop(redundant_features,axis=1)
    ##print feature_set.columns

    # 2. Univariate Feature Selection
    feature_set['stars'] = feature_set['stars'].apply(round) 
    availableFeatureSet = [col for col in feature_set.columns if col not in ['stars']]
    # feature selection
    kBest = SelectKBest(score_func=chi2, k=15)
    fit = kBest.fit(feature_set[availableFeatureSet], feature_set['stars'])
    # summarize scores
    np.set_printoptions(precision=3)
    ##print '\n',fit.scores_
    featuresSelected = fit.transform(feature_set[availableFeatureSet])
    # summarize selected features
    top10_indices = kBest.get_support(indices=True)
    ##print top10_indices
    remaining_columns = feature_set.columns
    top10_feature_names = [remaining_columns[idx] for idx, _  in enumerate(remaining_columns) if idx in top10_indices]
    print '\nSelected Top 15 Features using Univariate Selection :: ', top10_feature_names
    # The top 15 features are : ['review_count', 'stars', 'BikeParking', 'WheelchairAccessible', 'GoodForKids', 'HasTV', 'OutdoorSeating',  
    #                            'RestaurantsTakeOut', 'RestaurantsTableService', 'BusinessParkingTypes', 'RestaurantsPriceRange', 
    #                            'Ambience_Offered', 'Alcohol', 'Music_Offered', 'Categories_Offered']

    # 3. Recursive Feature Elimination
    model = LogisticRegression()
    rfe = RFE(model, 15)
    fit = rfe.fit(feature_set[availableFeatureSet], feature_set['stars'])
    ##print("\nFeature Ranking: %s") % fit.ranking_
    top10_indices_recursive = rfe.get_support(indices=True)
    top10_features_recursive = [remaining_columns[idx] for idx, _  in enumerate(remaining_columns) if idx in top10_indices_recursive]
    print '\nSelected Top 15 Features using Resursive Feature Elimination :: ', top10_features_recursive
    # The top 15 features are : ['stars', 'BikeParking', 'WheelchairAccessible', 'Caters', 'GoodForKids', 'HasTV', 'OutdoorSeating', 
    #                            'RestaurantsReservations', 'RestaurantsTakeOut', 'RestaurantsTableService', 'RestaurantsPriceRange',     
    #                            'Ambience_Offered', 'WiFi', 'Music_Offered', 'Categories_Offered']

    # 4. Tree Based Feature selection
    treeModel = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    treeModel.fit(feature_set[availableFeatureSet], feature_set['stars'])
    importances = treeModel.feature_importances_
    std = np.std([tree.feature_importances_ for tree in treeModel.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    top10_indices_tree = indices[:16]
    top10_features_tree = [remaining_columns[idx] for idx, _  in enumerate(remaining_columns) if idx in top10_indices_tree]
    print '\nSelected Top 10 Features using Extra Tree Classifier :: ', top10_features_tree
    # The top 10 features are :   ['review_count', 'stars', 'BikeParking', 'WheelchairAccessible', 'Caters', 'GoodForKids', 'HasTV', 
    #                              'OutdoorSeating', 'RestaurantsTakeOut', 'RestaurantsTableService', 'BusinessParkingTypes',    
    #                              'RestaurantsPriceRange', 'Ambience_Offered', 'Alcohol', 'WiFi', 'Music_Offered']

    # Selecting the features that were common amongst all classifiers
    top_10 = list(set(top10_feature_names) & set(top10_features_recursive) & set(top10_features_tree))
    if 'stars' in top_10:
        top_10 = top_10[0:11]
        top_10.remove('stars')
    else:
    	top_10 = top_10[0:10]
    print '\nTop 10 Features Identified by all classifiers :: ', top_10
    return top_10
    # All the above techniques produced the following 10 features in common::
    # ['RestaurantsTableService', 'HasTV', 'RestaurantsPriceRange', 'OutdoorSeating', 'Ambience_Offered', 'BikeParking', 'RestaurantsTakeOut', 
    #  'GoodForKids', 'WheelchairAccessible', 'Alcohol']
    # Hence we conclude that these are the top 10 features that are most relevant for predicting the restaurant performance/star rating

def buid_Naive_Bayes_Classifier(df):
   
    df['stars'] = df['stars'].apply(np.round)
    labels = df.loc[:,'stars']
    df_labels = labels.to_frame()
    df_attrs = df.drop(['stars'] , axis=1)
     
    gnb = GaussianNB()
    [tr_data, te_data, tr_labels, te_labels] = cross_validation.train_test_split(df_attrs, df_labels, test_size=0.2,random_state=42)
    gnb.fit(tr_data, tr_labels.values.ravel())

    # Get accuracy score
    print "Accuracy score :", gnb.score(te_data, te_labels)

    # K-Fold cross calidation
    foldnum = 0
    fold_results = pd.DataFrame()

    # Using 10-Fold cross validation
    for train, test in cross_validation.KFold(len(df_attrs), n_folds=10):
        foldnum+=1
        tr_data = pd.DataFrame(df_attrs).iloc[train]
        te_data = pd.DataFrame(df_attrs).iloc[test]
        tr_target = pd.DataFrame(df_labels).iloc[train]
        te_target = pd.DataFrame(df_labels).iloc[test]
        gnb = GaussianNB()
        gnb.fit(tr_data, tr_target.values.ravel())
        prob_arr_gnb = gnb.predict_proba(te_data)
        score_gnb = gnb.score(te_data, te_target)
        fold_results.loc[foldnum, 'Score'] = score_gnb

    print "Accuracy score across folds : \n", fold_results   
    # Get accuracy score accross folds    
    print "\nMean accuracy score :", fold_results.mean()
        
# Read the Nevada DataSet
df = pd.read_csv('NV_Restaurants.csv')
reduced_df = df.copy()
df.fillna(0)
# Remove the correlated features
feature_set = remove_correlated_features()
redundant_features = list(set(df.copy().columns.tolist()) - set(feature_set))
reduced_df = reduced_df.drop(redundant_features,axis=1)
# Find top 10 features that most relevant for predicting the restaurant performance/star rating
top_10_features = top_features(reduced_df)
top_10_features.append('stars')
other_features = list(set(reduced_df.columns.tolist()) - set(top_10_features))
reduced_df = reduced_df.drop(other_features,axis=1)

# Model building and comparison
# Total records : 7858, For testing training we use 80-20 ration. So training dataset will have 5500 records and testing dataset will have 
# 2358 records
# Note that for classification purpose, the response variable 'star' has been rounded to corresponding integer value
# 1. Multinomial Naive Bayes classifier
buid_Naive_Bayes_Classifier(reduced_df)

 