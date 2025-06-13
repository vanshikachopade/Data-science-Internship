import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Titanic dataset
df = pd.read_csv('train.csv')  # Make sure train.csv is in the same folder

# Step 2: Preview the data
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Check for missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Step 4: Data Cleaning
# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Step 5: Confirm missing values are handled
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Step 6: Exploratory Data Analysis (EDA)

# 1. Survival Count
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Number of Passengers')
plt.show()

# 2. Survival by Gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# 3. Age Distribution
sns.histplot(df['Age'], bins=20, kde=True, color='orange')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

# 4. Survival by Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# 5. Correlation Heatmap (Bonus)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()
