import pandas as pd
import requests # request img from web
import shutil # save img locally
import multiprocessing
from tqdm import tqdm 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--min", type=int)
parser.add_argument("--max", type=int)

args = parser.parse_args()
data_path = "D:\\CC_dataset\\Train_GCC-training (1).tsv"

nrows = args.max-args.min
if args.min < 0:
    df = pd.read_csv(data_path, sep='\t', names=['cap', 'url'], header=None
                 )
else: 
    df = pd.read_csv(data_path, sep='\t', names=['cap', 'url'], header=None,
                 nrows=nrows, skiprows=args.min)

def req_url(url, name):
    #url = row['url']
    file_name = 'train\\'+str(name)+'.jpg'
    res = requests.get(url, stream = True, timeout=2)
    if res.status_code == 200:
        with open(file_name,'wb') as f:
            shutil.copyfileobj(res.raw, f)
        #print('Image sucessfully Downloaded: ',file_name)
    else:
        #print('Image Couldn\'t be retrieved')
        pass

i = 0 

if args.min == 0 :
    values = df.iloc[args.min:args.max]['url'].values
    for row in tqdm(values):
        try: 
            req_url(row, args.min+i)
        except: 
            pass
        i+=1
elif args.min<0: 
    min = df.iloc[args.min:args.max].iloc[0].name
    for row in df.iloc[min:args.max]['url'].values:
        try: 
            req_url(row, min+i)
        except: 
            pass
        i+=1
else: 
    values = df.iloc[args.min:args.max]['url'].values
    del df
    for row in values:
        try: 
            req_url(row, args.min+i)
        except: 
            pass
        i+=1