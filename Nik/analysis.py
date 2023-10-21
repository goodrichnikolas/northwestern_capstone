import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd
import os
import glob

#Get all csvs from the csv_output folder
path = r'./Nik/csv_output'
csvs = glob.glob(os.path.join(path, "*.csv"))

#Select most recent csv
csv = max(csvs, key=os.path.getctime)

#Read csv
df = pd.read_csv(csv)

#Create chart
plt.plot(df['number_of_games'], df['scores'])
plt.title('Lunar Lander')
plt.xlabel('Nummber of Games')
plt.ylabel('Scores')
plt.show()

#Save chart
plt.savefig('lunarlander.png')
