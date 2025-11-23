# Run garbage collection to free memory
import gc
gc.collect()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, 
                             roc_curve, auc, ConfusionMatrixDisplay,precision_score, recall_score) 
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv("RideHailingData_100k.csv")

print(df.info())
print(df.dtypes)
# Display the summary statistics of the dataset
print(df.describe(include='all'))

# Convert 'CityZone' to categorical variable
df['CityZone'] = df['CityZone'].astype('category')
print(df.dtypes)

##############################
#1. Create a Binary Target Variable (Profitable vs Not)
# Create ProfitableRider (1 = profitable, 0 = not profitable)
df["ProfitableRider"] = (df["RiderProfit"] >= 0).astype(int)
# Check class distribution
class_dist = df["ProfitableRider"].value_counts(normalize=True) * 100
print("Class Distribution (%):\n", class_dist)

##############################
#2. Handle Missing Values (Age and Income)
# Impute missing values with median (ordinal variables) 
for col in ["RiderAgeGroup", "RiderIncomeGroup"]:
    # Impute with median
    median_val = df[col].median()
    df.fillna({col:median_val},inplace=True)

# Verify no remaining missing values
print("Missing values after imputation:\n", df[["RiderAgeGroup", "RiderIncomeGroup"]].isna().sum())

##############################
#3. Train-Test Split (75%/25%)
# Define features (X) and target (y)
X = df.drop(["RiderProfit", "ProfitableRider"], axis=1)
y = df["ProfitableRider"]

# Split data (stratified for class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")

##############################
#4.a Model Training – Logistic, Tree, Random Forest

# 1）Logistic Regerssion
import statsmodels.api as sm
X_train = pd.get_dummies(X_train, columns=["CityZone"], drop_first=True, dtype=float)
X_train_const = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit()
# Print the summary of the model
print(result.summary())

#4.b Decision Tree (with Pruning Tuning)
# Hyperparameter tuning (avoid overfitting)
dt_params = {"max_depth": [3, 5, 7, 10], "min_samples_leaf": [5, 10, 20]}
dt = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(dt, dt_params, cv=5, scoring="roc_auc")
dt_grid.fit(X_train, y_train)

# Best decision tree model
best_dt = dt_grid.best_estimator_
print(f"Best Decision Tree Params: {dt_grid.best_params_}")

#4.c Random Forest (with Hyperparameter tuning)
rf_params = {"n_estimators": [100, 200], "max_depth": [5, 10], "max_features": ["sqrt"]}
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="roc_auc")
rf_grid.fit(X_train, y_train)

# Best random forest model
best_rf = rf_grid.best_estimator_
print(f"Best Random Forest Params: {rf_grid.best_params_}")

##############################
#5. Model Evaluation (Accuracy, Confusion Matrix, ROC/AUC)

#1) Evaluate Logistic Regression 

# Assuming X_test, y_test are defined and preprocessed similarly to X_train (including dummy encoding for CityZone)
# Also assuming y_pred_proba and y_pred are obtained from the model's prediction
# First, get predictions and predicted probabilities for the test set
X_test = pd.get_dummies(X_test, columns=["CityZone"], drop_first=True, dtype=float)
# Ensure X_test has the same columns as X_train_const (excluding the constant added by sm.add_constant)

# Add constant for prediction
X_test_const = sm.add_constant(X_test)
# Predict probabilities
y_pred_proba = result.predict(X_test_const)
# Predict classes (using a threshold, default is 0.5)
y_pred = (y_pred_proba > 0.5).astype(int)

# a. Calculate Accuracy on training set and test set
# Training set prediction
y_train_pred_proba = result.predict(X_train_const)
y_train_pred = (y_train_pred_proba > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# b. Generate Confusion Matrix for test set
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Test Set):")
print(conf_matrix)

# Create a heatmap for the confusion matrix with seaborn
plt.figure(figsize=(6, 4)) # Create a heatmap for the confusion matrix with labels
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted Negative', 'Predicted Positive'],
yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix-Logistic Regression')
plt.show()

# c. Plot ROC Curve and calculate AUC for test set
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

print(f"AUC (Test Set): {roc_auc:.4f}")

##############################
#2) Evaluate Decision Tree 
#a.Calculate accuracy score
train_accuracy_tree = best_dt.score(X_train, y_train)
test_accuracy_tree = best_dt.score(X_test, y_test)

print("\nDT_Train_Accuracy：", train_accuracy_tree * 100, "%")
print("\nDT_test_Accuracy：", test_accuracy_tree * 100, "%")

tree_predictions = best_dt.predict(X_test)
tree_conf_matrix = confusion_matrix(y_test, tree_predictions)
print("\nDT_Confusion_Matrix：")
print(tree_conf_matrix)

#b. Plot Confusion Matrix
plt.figure(figsize=(6, 4)) # Create a heatmap for the confusion matrix with labels
sns.heatmap(tree_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted Negative', 'Predicted Positive'],
yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix-Decision Tree')
plt.show()

#c. Plot ROC Curve and calculate AUC for test set
y_pred_proba = best_dt.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Decision Tree')
plt.legend(loc="lower right")
plt.show()
print(f"Decision Tree AUC: {roc_auc:.3f}")

##############################
#3) Evaluate Random Forest
#a.Calculate accuracy score
train_accuracy_forest = best_rf.score(X_train, y_train)
test_accuracy_forest = best_rf.score(X_test, y_test)
print(f"\nRF_Train_Accuracy：{train_accuracy_forest * 100:.2f}%")
print(f"RF_Test_Accuracy：{test_accuracy_forest * 100:.2f}%")

forest_predictions = best_rf.predict(X_test)
forest_conf_matrix = confusion_matrix(y_test, forest_predictions)
print("\nRT_Confusion_Matrix：")
print(forest_conf_matrix)

#b. Plot Confusion Matrix
plt.figure(figsize=(6, 4)) # Create a heatmap for the confusion matrix with labels
sns.heatmap(forest_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted Negative', 'Predicted Positive'],
yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix-Random Forest')
plt.show()

#c. Plot ROC Curve and calculate AUC for test set
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Random Forest')
plt.legend(loc="lower right")
plt.show()
print(f"Random Forest AUC: {roc_auc:.3f}")

##############################
#6.Classification Threshold Analysis

# Use best model (Random Forest) for threshold analysis
best_model = best_rf
rf_y_test_prob = best_rf.predict_proba(X_test)

# Check number of classes
n_classes = rf_y_test_prob.shape[1]
print(f"Detected {n_classes} classes")

# Analyze different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6]
threshold_results = []

for thresh in thresholds:
    # Predict based on threshold
    if n_classes == 2:
        y_pred = (rf_y_test_prob[:, 1] >= thresh).astype(int)
    else:
        max_probs = rf_y_test_prob.max(axis=1)
        y_pred = np.where(max_probs >= thresh, rf_y_test_prob.argmax(axis=1), -1)
    
    # Calculate metrics
    avg = 'binary' if n_classes == 2 else 'weighted'
    prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
    
    threshold_results.append({"Threshold": thresh, "Precision": prec, "Recall": rec})
    print(f"\nThreshold = {thresh}: Precision = {prec:.3f}, Recall = {rec:.3f}")

# Create DataFrame
df_threhold = pd.DataFrame(threshold_results)
print("\nThreshold Summary:\n", df_threhold)

# Plot precision-recall curve
plt.figure(figsize=(10, 6))
plt.plot(df_threhold['Recall'], df_threhold['Precision'], 'o-', color='b', label='PR Curve')

# Add threshold labels
for i, thresh in enumerate(df_threhold['Threshold']):
    plt.annotate(f'T={thresh}', (df_threhold['Recall'][i], df_threhold['Precision'][i]),
                 xytext=(0,10), textcoords="offset points", ha='center')

plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xlim(0.6, 1.05)
plt.ylim(0.6, 0.8)
plt.tight_layout()
plt.show()

##############################
# 7. Cost-Benefit Simulation
TP_BENEFIT = 50  # Profit per True Positive
FP_COST = 10      # Cost per False Positive (promotion)
FN_LOSS = 50      # Loss per missed profitable rider (False Negative)

def calculate_total_benefit(thresh, y_true, y_prob, n_classes):
    # Generate predictions
    if n_classes == 2:
        y_pred = (y_prob[:, 1] >= thresh).astype(int)
    else:
        max_probs = y_prob.max(axis=1)
        y_pred = np.where(max_probs >= thresh, y_prob.argmax(axis=1), n_classes)
    
    # Extract TP, FP, FN
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)[1] if n_classes == 2 else cm[1, 1]
    fp = cm.sum(axis=0)[1] - tp
    fn = cm.sum(axis=1)[1] - tp
    
    # Calculate benefit
    return (tp * TP_BENEFIT) - (fp * FP_COST) - (fn * FN_LOSS)

# Evaluate thresholds
cost_benefit_results = [{"Threshold": t, "Total Benefit": calculate_total_benefit(t, y_test, rf_y_test_prob, n_classes)} for t in df_threhold['Threshold']]
cost_benefit_df = pd.DataFrame(cost_benefit_results)

# Print results
for _, row in cost_benefit_df.iterrows():
    print(f"Threshold = {row['Threshold']}: Total Benefit = ${row['Total Benefit']:,}")
print("\nCost-Benefit Summary:\n", cost_benefit_df)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(cost_benefit_df['Threshold'], cost_benefit_df['Total Benefit'], 'o-', color='g', label='Total Benefit')
plt.title('Total Benefit vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Total Benefit ($)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()























