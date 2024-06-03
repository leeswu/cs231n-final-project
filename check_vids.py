import os

# Define the path to the file list and the directory to check
test_file_list_path = 'data/k700-2020/updated_splits/small_test.txt'
val_directory_path = 'data/k700-2020/updated_splits/small_val'

# Read the list of file names from small_test.txt
with open(test_file_list_path, 'r') as file:
    test_files = [line.strip() for line in file]

# Check if each file in the list exists in the small_val directory
missing_files = []
for test_file in test_files:
    if not os.path.isfile(os.path.join(val_directory_path, test_file)):
        missing_files.append(test_file)

# Output the results
if missing_files:
    print(f"The following files listed in {test_file_list_path} are missing in the {val_directory_path} directory:")
    for missing_file in missing_files:
        print(missing_file)
else:
    print(f"All files listed in {test_file_list_path} are present in the {val_directory_path} directory.")
