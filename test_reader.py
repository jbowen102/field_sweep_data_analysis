import os
import csv

file_path = input("Enter file_path:\n> ")

if os.path.splitext(file_path)[1].lower() == ".tsv":
    # TSV export contains NUL bytes at the end that causes exception in csv.reader().
    # Remove
    with open(file_path, 'rb') as fi:
        data = fi.read()
    cleaned_file_path = os.path.splitext(file_path)[0] + "_cleaned" + os.path.splitext(file_path)[1]
    with open(cleaned_file_path, "wb") as fo:
        fo.write(data.replace(b'\x00', b''))
    # https://stackoverflow.com/questions/4166070/python-csv-error-line-contains-null-byte
else:
    # ASCII file doesn't require above cleaning.
    cleaned_file_path = file_path

ecu_file_obj = open(cleaned_file_path, "r")
ecu_data = csv.reader(ecu_file_obj, delimiter="\t")

for i, row in enumerate(ecu_data):
    if i < 10:
        print(row)

# row_length = 42
# for i, row in enumerate(ecu_data):
#     if len(row) != row_length:
#         print("%d: %s" % (i, row))

print("\n%s: %d" % ("row count", i))
