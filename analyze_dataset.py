import pandas as pd

# Load the new dataset
file_path = "Hackathon_Round_2.xlsx"
df = pd.read_excel(file_path)

# Display dataset structure
print("\n📊 First 5 Rows:")
print(df.head())

print("\n📌 Dataset Info:")
print(df.info())

print("\n🔍 Missing Values:")
print(df.isnull().sum())
