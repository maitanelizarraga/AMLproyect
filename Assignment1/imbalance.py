

def adjustingweights(df): 
    #as we have seen, the dataset is imbalanced, so we will adjust the weights of the classes to balance them
    from sklearn.utils import class_weight
    import numpy as np
    class_weights = class_weight.compute_class_weight('balanced', np.unique(df['is_fraud']), df['is_fraud'])
    print("Class weights: ", class_weights)

def automaticweights(df): 
    #we can also use the automatic weights of the models, which will adjust the weights of the classes automatically
    print("Automatic weights: ", "balanced")

def comparison(df): 
    #we can compare the results of the models with and without adjusting the weights, and with automatic weights, to see which one performs better
    print("Comparison of models with different weight adjustments: ")
    print("Model with adjusted weights: ")
    adjustingweights(df)
    print("Model with automatic weights: ")
    automaticweights(df)

def oversamplongsmote(df): 
    #we can also use oversampling techniques, such as SMOTE, to balance the dataset
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print("Original dataset shape: ", X.shape)
    print("Resampled dataset shape: ", X_resampled.shape)

def oversamplingadasyn(df): 
    #we can also use the ADASYN oversampling technique, which is similar to SMOTE but focuses on harder-to-learn examples
    from imblearn.over_sampling import ADASYN
    from sklearn.model_selection import train_test_split
    
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    
    print("Original dataset shape: ", X.shape)
    print("Resampled dataset shape: ", X_resampled.shape)


def oversamplingrandom(df): 
    #we can also use random oversampling, which simply duplicates examples from the minority class to balance the dataset
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.model_selection import train_test_split
    
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    print("Original dataset shape: ", X.shape)
    print("Resampled dataset shape: ", X_resampled.shape)

def undersamplingnearmiss(df): 
    #we can also use the NearMiss undersampling technique, which selects examples from the majority class that are closest to the minority class examples
    from imblearn.under_sampling import NearMiss
    from sklearn.model_selection import train_test_split
    
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    
    nearmiss = NearMiss()
    X_resampled, y_resampled = nearmiss.fit_resample(X, y)
    
    print("Original dataset shape: ", X.shape)
    print("Resampled dataset shape: ", X_resampled.shape)


def undersamplingtomek(df):
    #we can also use the Tomek Links undersampling technique, which removes examples from the majority class that are close to the minority class examples
    from imblearn.under_sampling import TomekLinks
    from sklearn.model_selection import train_test_split
    
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    
    tomek = TomekLinks()
    X_resampled, y_resampled = tomek.fit_resample(X, y)
    
    print("Original dataset shape: ", X.shape)
    print("Resampled dataset shape: ", X_resampled.shape)



def undersamplingrandom(df): 
    #we can also use random undersampling, which simply removes examples from the majority class to balance the dataset
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.model_selection import train_test_split
    
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    print("Original dataset shape: ", X.shape)
    print("Resampled dataset shape: ", X_resampled.shape)

def balancedrandomforest(df): 
    #we can also use the BalancedRandomForestClassifier, which is a random forest classifier that adjusts the weights of the classes automatically
    from imblearn.ensemble import BalancedRandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    
    brf = BalancedRandomForestClassifier(random_state=42)
    brf.fit(X, y)
    
    print("Balanced Random Forest Classifier fitted successfully")