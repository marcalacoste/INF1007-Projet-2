import pandas as pd
import numpy as np
import os
import csv

path = os.path.join("donnees", "data.csv")

data = pd.read_csv(path)

# print(data.isnull().sum())

with open (path, "r", encoding = "utf8") as f:
    with open("donnees.csv", "w", encoding = "utf8", newline = "") as f2:
        f = csv.reader(f)
        f2 = csv.writer(f2)
        for line in f:
            if line[2] == "":
                continue
            f2.writerow(line)
