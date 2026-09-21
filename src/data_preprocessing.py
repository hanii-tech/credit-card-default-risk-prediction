import pandas as pd

# Load dataset
file_path = "data/default of credit card clients.xls"
df = pd.read_excel(file_path, header=1)

# Rename target column
df.rename(columns={"default payment next month": "default"}, inplace=True)

# Remove customer ID
df.drop("ID", axis=1, inplace=True)

# Handle unknown education codes
df["EDUCATION"] = df["EDUCATION"].replace({
    0: 4,
    5: 4,
    6: 4
})

# Handle unknown marriage code
df["MARRIAGE"] = df["MARRIAGE"].replace({
    0: 3
})

# Convert categorical columns into dummy variables
df = pd.get_dummies(
    df,
    columns=["SEX", "EDUCATION", "MARRIAGE"],
    drop_first=True
)

# Save cleaned dataset
output_path = "data/cleaned_credit_card_data.csv"
df.to_csv(output_path, index=False)

# Display results
print("Preprocessing completed successfully!")
print("\nCleaned dataset shape:")
print(df.shape)

print("\nCleaned columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())