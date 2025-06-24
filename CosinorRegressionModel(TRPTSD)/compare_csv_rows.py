import pandas as pd

# Load the two CSV files
pre_df = pd.read_csv('CosinorRegressionModel(TRPTSD)/data/RNS_G_Pre_output.csv')
post_df = pd.read_csv('CosinorRegressionModel(TRPTSD)/data/RNS_G_M1_output.csv')

# Ensure the same number of rows
min_len = min(len(pre_df), len(post_df))
pre_df = pre_df.iloc[:min_len]
post_df = post_df.iloc[:min_len]

# Compare each row
not_identical = []
for i in range(min_len):
    if not pre_df.iloc[i].equals(post_df.iloc[i]):
        not_identical.append(i)

# Print the indices (and optionally the timestamps) of differing rows
print(f"Rows that are not identical (total {len(not_identical)}):")
if not not_identical:
    print("All rows are identical between the two files.")
else:
    for idx in not_identical:
        print(f"Row {idx}: Pre = {pre_df.iloc[idx].to_dict()}, Post = {post_df.iloc[idx].to_dict()}") 