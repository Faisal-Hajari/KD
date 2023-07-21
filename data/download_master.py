import subprocess
import pandas as pd 
data_path = "D:\\CC_dataset\\Train_GCC-training (1).tsv"
df = pd.read_csv(data_path, sep='\t', names=['cap', 'url'], header=None)

size = len(df)


n_process = 500
step = int(size/n_process)
print(f"num_process: {n_process} | chunck_size: {step} | total_size: {size}")

for i in range(0, size, step): 
    process = subprocess.Popen(['python', 'download.py', '--min',str(i) ,'--max',str(i+step)])


process = subprocess.Popen(['python', 'download.py', '--min',str(-step) ,'--max',str(size)])
