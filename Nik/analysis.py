import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd

df_0 = pd.read_csv('./Nik/lunarlander_eps_0.0.csv')
df_1 = pd.read_csv('./Nik/lunarlander.csv')

#compare the scores
plt.figure()
plt.plot(df_0['scores'], label='eps=0.0')
plt.plot(df_1['scores'], label='eps=1.0')
#x axis
plt.xlabel('Number of games')
#y axis
plt.ylabel('Scores')
plt.legend()
plt.savefig('./Nik/scores.png')

