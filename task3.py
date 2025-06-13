import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# Load dataset
df = pd.read_csv("bank-full.csv", sep=';')

# Display basic info
print("First 5 rows:")
print(df.head())

# Encode categorical variables
label_encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# Split into features (X) and target (y)
X = df.drop("y", axis=1)
y = df["y"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the tree (small depth for clarity)
plt.figure(figsize=(20,10))
tree.plot_tree(clf, max_depth=2, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.title("Decision Tree Overview (First 2 Levels)")
plt.show()

