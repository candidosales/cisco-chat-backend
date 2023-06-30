import pandas as pd


# Load your CSV file
df = pd.read_csv(
    "./data/output-2023-06-22T03:37:39.362Z.csv"
)  # replace 'file.csv' with your file path

# Select rows with duplicate values in a specific column
duplicates = df[
    df.duplicated(subset="url", keep=False)
]  # replace 'column_name' with your column

# Print the duplicates
print(duplicates)
