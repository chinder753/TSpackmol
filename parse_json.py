import json

with open("PeriodicTableJSON.json", encoding='utf-8') as file:
    ele = json.load(file)["elements"]

for x in ele:
    print("'"+x["symbol"]+"'", end=",")