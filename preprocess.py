import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/oasis_longitudinal.csv")

# Use only baseline visits
df = df[df['MRI ID'].str.contains('MR1')]

# ✅ Filter only Demented and Nondemented
df = df[df['Group'].isin(['Demented', 'Nondemented'])]

# Encode 'Group' column: Demented = 1, Nondemented = 0
df['Group'] = df['Group'].replace({'Demented': 1, 'Nondemented': 0})

# Encode gender
df['M/F'] = df['M/F'].replace({'M': 1, 'F': 0})

# Drop unnecessary columns
df = df.drop(columns=['Subject ID', 'MRI ID', 'Visit', 'Hand'])

# Drop rows with missing values
df = df.dropna()

# Save processed file
df.to_csv("data/processed_data.csv", index=False)
print("✅ Preprocessing complete.")


