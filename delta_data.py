import pandas as pd
import csv
import numpy as np

mean = pd.read_csv('data.csv', sep='\t')
mean_base = pd.read_csv('delta_base.csv')
result = []
indx = ["Fascination","Being_away_from","Negative", "Valence", "Arousal"]
for idx in indx:
    print(idx)  
    tmp = []
    for i in range(len(mean)):
        fl = 0
        for j in range(len(mean_base)):
            if mean["ID"][i] == mean_base["ID"][j]:
                tmp.append(mean[idx][i]-mean_base[idx][j])
                fl = 1
                break
        if fl == 0 :
                tmp.append(-100) 
    print(len(tmp))
    result.append(tmp)

numpy_array = np.array(result)
transpose = numpy_array.T
transpose_list = transpose.tolist()

with open('shows.csv', 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerows(transpose_list)