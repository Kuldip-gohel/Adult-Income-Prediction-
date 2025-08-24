import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the data
adult = pd.read_csv("Adult_data_set.csv")

# 2. Create a stratified test set 
adult["education_cat"] = pd.cut(
    adult["education-num"],
    bins=[0,5,10,15,20, np.inf],
    labels=[1,2,3,4,5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_data, test_data in split.split(adult, adult["education_cat"]):
    train_set = adult.loc[train_data].drop("education_cat",axis=1)
    test_set = adult.loc[test_data].drop("education_cat",axis=1)

# 3. Now work on a copy of training data
adult = train_set.copy()

# 4. Lets Separate labels and features
adult_labels = adult["salary"].copy()
adult = adult.drop("salary", axis=1)

# 5. seperate numerical and categorical columns
num_attribs = adult.drop(["workclass", "education", "occupation", "sex", "native-country"], axis=1).columns.tolist()
cat_attribs = ["workclass", "education", "occupation", "sex", "native-country"]

# 6. create pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
 
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# 7. Transform the data
adult_prepared = full_pipeline.fit_transform(adult)

# 8. Train Random Forest Classifier
random_forest_clf = RandomForestClassifier(random_state=42)
random_forest_clf.fit(adult_prepared, adult_labels)

# Predictions
rf_preds = random_forest_clf.predict(adult_prepared)

# Accuracy on training set
train_acc = accuracy_score(adult_labels, rf_preds)
print(f"Training Accuracy: {train_acc}")

# Cross-validation Accuracy
rf_cv_scores = cross_val_score(random_forest_clf, adult_prepared, adult_labels, cv=10, scoring="accuracy")
print("Cross-validation accuracy stats:")
print(pd.Series(rf_cv_scores).describe())






# Now i check this model in Test_set.
adult_test = test_set.copy()

# Separate labels and features
adult_test_labels = adult_test["salary"].copy()
adult_test = adult_test.drop("salary", axis=1)

# Apply the same transformations as above
adult_test_prepared = full_pipeline.transform(adult_test)

# Evaluate on test data
test_preds = random_forest_clf.predict(adult_test_prepared)

# check the Accuracy of model
test_acc = accuracy_score(adult_test_labels, test_preds)
print(f"Test Accuracy: {test_acc}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(adult_test_labels, test_preds))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(adult_test_labels, test_preds))
