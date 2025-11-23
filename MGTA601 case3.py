import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.feature_selection import SelectKBest, f_classif

# ---------------------- 1. Data Loading & Initial Exploration ----------------------
df = pd.read_csv('mushrooms.csv')

print("Dataset shape (rows, columns):", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nClass distribution (e=edible, p=poisonous):")
print(df['class'].value_counts())

print("\nMissing values (marked as '?') per feature:")
print(df.isin(['?']).sum())

# ---------------------- 2. Data Preprocessing ----------------------
X = df.drop('class', axis=1)
y = df['class']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # e→0, p→1

# One-hot encode categorical features
categorical_features = X.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_features)
    ],
    remainder='drop'
)

# Split into training and testing sets (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------------- 3. Model Training, Evaluation & Visualization ----------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

model_results = {}
roc_results = {}

# 1. ROC Curve Visualization
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selector', SelectKBest(f_classif, k=20)),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of 'poisonous (p)'
    
    # Calculate ROC metrics
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    roc_results[model_name] = (fpr, tpr, auc_roc)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_roc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve - Mushroom Toxicity Prediction')
plt.legend()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Model Metrics Comparison Visualization
metrics_df = []
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selector', SelectKBest(f_classif, k=20)),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    metrics_df.append({
        'Model': name,
        'Accuracy': accuracy,
        'AUC-ROC': auc_roc
    })

metrics_df = pd.DataFrame(metrics_df)
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(metrics_df))
plt.bar(index - bar_width/2, metrics_df['Accuracy'], bar_width, label='Accuracy')
plt.bar(index + bar_width/2, metrics_df['AUC-ROC'], bar_width, label='AUC-ROC')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(index, metrics_df['Model'])
plt.legend()
plt.savefig('model_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Determine Best Model and Output Detailed Results
best_model_idx = metrics_df[['Accuracy', 'AUC-ROC']].apply(tuple, axis=1).argmax()
best_model_name = metrics_df.loc[best_model_idx, 'Model']
best_pipeline = None

for name, model in models.items():
    if name == best_model_name:
        best_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selector', SelectKBest(f_classif, k=20)),
            ('classifier', model)
        ])
        best_pipeline.fit(X_train, y_train)
        break

print("\n=== Detailed Model Results ===")
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selector', SelectKBest(f_classif, k=20)),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]):.4f}")
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Edible (e)', 'Poisonous (p)']))

print(f"\nBest Model: {best_model_name}")


# ---------------------- 4. Feature Importance Visualization (Only for Best Model) ----------------------
if best_model_name in ["Decision Tree", "Random Forest"] and best_pipeline is not None:
    ohe = preprocessor.named_transformers_['cat']
    encoded_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_selector = best_pipeline.named_steps['feature_selector']
    selected_indices = feature_selector.get_support(indices=True)
    selected_features = encoded_feature_names[selected_indices]
    
    if best_model_name == "Random Forest":
        rf = best_pipeline.named_steps['classifier']
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
    else:  # Decision Tree
        dt = best_pipeline.named_steps['classifier']
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': dt.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.xlabel('Importance Score')
    plt.title(f'Best Model ({best_model_name}) - Top 10 Important Features')
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------- 5. Confusion Matrix Heatmap Visualization (Only for Best Model) ----------------------
y_pred_best = best_pipeline.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Edible (e)', 'Poisonous (p)'],
            yticklabels=['Edible (e)', 'Poisonous (p)'])
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Best Model ({best_model_name}) - Confusion Matrix Heatmap')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

