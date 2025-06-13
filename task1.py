import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dummy dataset
data = pd.DataFrame({
    'Name': ['Amit', 'Sara', 'Ravi', 'Anjali', 'Vikram', 'Priya', 'John', 'Kavya'],
    'Age': [22, 25, 21, 24, 23, 26, 28, 22],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female']
})

# --- Bar Chart: Gender Distribution ---
gender_counts = data['Gender'].value_counts()

gender_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Histogram: Age Distribution ---
plt.hist(data['Age'], bins=5, color='orange', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
