import csv
import pandas as pd

# Open the CSV file in read mode
with open('/home/cv/Desktop/KD/CLIPPO/test.csv', 'r') as csv_file:
    mycsv = csv.reader(csv_file)
    mycsv = list(mycsv)
    text = mycsv[1][6]
    print(text)
