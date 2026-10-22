import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    "Income": [50000, 30000, 60000, 25000, 80000,
               35000, 70000, 20000, 90000, 40000],
    "Credit_History": [1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    "Loan_Amount": [200000, 150000, 250000, 100000,
                    300000, 120000, 280000, 90000,
                    350000, 160000],
    "Loan_Status": ["Approved", "Rejected", "Approved",
                    "Rejected", "Approved", "Approved",
                    "Approved", "Rejected", "Approved",
                    "Rejected"]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Income", "Credit_History", "Loan_Amount"]]
y = df["Loan_Status"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

# Train Machine Learning model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Loan Approval Prediction System")
print("--------------------------------")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Get applicant details
print("\nEnter Applicant Details")

income = float(input("Monthly Income: "))
credit = int(input("Credit History (1 = Good, 0 = Poor): "))
loan = float(input("Loan Amount: "))

# Make prediction
applicant = pd.DataFrame(
    [[income, credit, loan]],
    columns=["Income", "Credit_History", "Loan_Amount"]
)

prediction = model.predict(applicant)[0]

print("\nPrediction Result:")
if prediction == "Approved":
    print("Loan Status: Approved")
else:
    print("Loan Status: Rejected")
