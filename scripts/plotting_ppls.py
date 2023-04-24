import pandas as pd
from pathlib import Path

p = Path(__file__).with_name('perplexities.txt')
with open(p, "r") as file:
    lines = file.readlines()
    file_table = list()

file_table = [line.strip().split() for line in lines]

table = pd.Series(file_table)
print(table)
