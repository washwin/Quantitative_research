import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

# Sample DataFrame
# data = pd.DataFrame({
#     'customer_id': [1, 2, 3, 4, 5],
#     'credit_lines_outstanding': [3, 5, 2, 4, 6],
#     'loan_amt_outstanding': [10000, 20000, 15000, 5000, 25000],
#     'total_debt_outstanding': [15000, 25000, 20000, 10000, 30000],
#     'income': [50000, 75000, 60000, 40000, 90000],
#     'years_employed': [5, 10, 7, 3, 15],
#     'fico_score': [700, 650, 720, 680, 630],
#     'default': [0, 1, 0, 0, 1]
# })

data = pd.read_csv("Loan_Data.csv")
# Features and target
X = data.drop(['customer_id', 'default'], axis=1)
y = data['default']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {auc}")
print("Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# Function to predict PD and expected loss
def calculate_expected_loss(features, loan_amount, model, scaler):
    features_scaled = scaler.transform([features])
    pd = model.predict_proba(features_scaled)[:, 1][0]
    recovery_rate = 0.1
    expected_loss = pd * (1 - recovery_rate) * loan_amount
    return expected_loss

# Example usage
example_borrower = X.iloc[0].values  # Replace with new borrower details
example_loan_amount = 10000  # Replace with the loan amount
loss = calculate_expected_loss(example_borrower, example_loan_amount, model, scaler)
print(f"Expected Loss: {loss}")
