import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load and Inspect Data (Booking ID is "row#")
try:
    df = pd.read_csv("your_data.csv")  
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Raw data file not found. Please check the file path.")
    exit()

if "row#" not in df.columns:
    raise ValueError("Missing booking ID column! Your data uses 'row#' as the booking ID.")

# Step 2: Initial Data Inspection
print("\n=== Initial Data Overview ===")
print(f"Total rows: {len(df)} | Total columns: {len(df.columns)}")
required_fields = ["row#", "user_id", "travel_type_id", "package_id", "from_date", "booking_created", "Car_Cancellation"]
for field in required_fields:
    if field in df.columns:
        print(f"✓ {field}: Data type = {df[field].dtype}, Non-null count = {df[field].notna().sum()}")
    else:
        print(f"✗ {field}: Missing")

# Step 3: Data Cleaning
print("\n=== Starting Data Cleaning ===")

# 3.1 Remove Duplicates (using "row#" as booking ID)
initial_rows = len(df)
df = df.drop_duplicates(subset=["row#"], keep="first")
print(f"Removed {initial_rows - len(df)} duplicate booking(s).")

# 3.2 Handle Missing Values for to_area_id (hourly rental logic)
if "to_area_id" in df.columns:
    df["to_area_id"] = df.apply(
        lambda row: "Valid: No Fixed Destination" 
        if (pd.isna(row["to_area_id"]) and row["travel_type_id"] == 3) 
        else row["to_area_id"],
        axis=1
    )
    df["to_area_id"] = df["to_area_id"].fillna("Missing: Invalid")

# 3.3 Convert Date Fields to Datetime
date_fields = ["from_date", "booking_created", "to_date"]
for field in date_fields:
    if field in df.columns:
        df[field] = pd.to_datetime(df[field], errors="coerce")
        invalid_count = df[field].isna().sum()
        if invalid_count > 0:
            df = df.dropna(subset=[field])
            print(f"Removed {invalid_count} row(s) with invalid {field}.")

# 3.4 Create Derived Fields
# Booking Lead Time (hours)
df["booking_lead_time_hour"] = (df["from_date"] - df["booking_created"]).dt.total_seconds() / 3600
df = df[df["booking_lead_time_hour"] >= 0]

# Trip Peak Period
def classify_peak_period(hour):
    if 6 <= hour <= 9:
        return "Morning Peak"
    elif 17 <= hour <= 20:
        return "Evening Peak"
    else:
        return "Off-Peak"
df["trip_peak_period"] = df["from_date"].dt.hour.apply(classify_peak_period)

# Standardize Package Type
package_mapping = {
    1: "4hrs-40kms", 2: "8hrs-80kms", 3: "6hrs-60kms",
    4: "10hrs-100kms", 5: "5hrs-50kms", 6: "3hrs-30kms",
    7: "12hrs-120kms"
}
if "package_id" in df.columns:
    df["package_type"] = df["package_id"].map(package_mapping).fillna("Unknown")

# 3.5 Filter Invalid Car_Cancellation Values
if "Car_Cancellation" in df.columns:
    df = df[df["Car_Cancellation"].isin([0, 1])]

# Step 4: Feature Engineering for Modeling
# Identify and Encode All Categorical Variables
categorical_cols = [
    "travel_type_id", "package_type", "trip_peak_period", 
    "online_booking", "mobile_site_booking", "to_area_id"
]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define Features (X) and Target (y)
X = df_encoded.drop(["row#", "Car_Cancellation", "from_date", "booking_created", "to_date"], axis=1)
y = df_encoded["Car_Cancellation"]

# Step 5: Handle Missing Values with Imputation
# Create a pipeline to impute missing values (using mean for numerical features)
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)  # Convert back to DataFrame

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {len(X_train)} | Testing set size: {len(X_test)}")

# Step 6: Build and Evaluate Models
## Model 1: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

## Model 2: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Evaluate Models
print("\n=== Model Evaluation ===")
## Random Forest
print("\nRandom Forest Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

## Logistic Regression
print("\nLogistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Not Cancelled", "Cancelled"], 
            yticklabels=["Not Cancelled", "Cancelled"])
plt.title("Random Forest: Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix_rf.png")
plt.close()

# Feature Importance from Random Forest
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance)
plt.title("Top 10 Features by Importance (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Step 7: Export Cleaned Data for Tableau
cleaned_file_path = "cleaned_taxi_booking_data.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned data exported to: {cleaned_file_path}")