import argparse
import commands

import sys
import os

import numpy as np
import pandas as pd

from multiprocessing import Pool
import itertools


n_processed = 0
n_total = 0


def oracle(x):
    if len(x) != 10:
        raise NameError("Length of input should be equal to 10")
    else:
        query = "java -cp OracleRegression.jar Oracle "
        query += " ".join(map(str, x))
        result = float(commands.getoutput(query))
        
        global n_processed
        n_processed += 1
        if n_processed % 10 == 0:
            print "{}:$ {} of {}".format(os.getpid(), n_processed, n_total)
        
        return result
        

def parse_args():
    parser = argparse.ArgumentParser(description="Oracle Manager")
    parser.add_argument('-n', '--number', action="store", type=int, default=200, help="Number of splits")
    parser.add_argument('-fin',  '--fin',  action="store", help="File to split")
    parser.add_argument('-fout', '--fout', action="store", help="File to save")
    return parser.parse_args()


def process_batch(b):
    global n_total    
    x = pd.read_csv(args.fin, sep=',', nrows=b[1], skiprows=b[0]).values
    n_total = len(x)
    return [(int(x[i, 0]), oracle(x[i, 1:])) for i in xrange(n_total)]
    

def process(n_lines, n_processes):
    if n_lines < 1:
        raise NameError("No rows found")
    if n_lines < n_processes:
        raise NameError("Number of processes is greater than number of lines")
    
    pool = Pool(processes=n_processes)
    
    split_size = n_lines / n_processes
    split = [(i, split_size if i + split_size < n_lines else n_lines - i)
                for i in xrange(0, n_lines, split_size)]
    
    y = pool.map(process_batch, split)
    y = list(itertools.chain(*y))
    
    pool.close()
    pool.join()
    
    return y
    

if __name__ == "__main__":
    args = parse_args()
    
    
    n_lines = -1
    with open(args.fin, "r") as f_name:
        for line in f_name:
            n_lines += 1
            
    # X_train = pd.read_csv(args.fin, sep=',').values
    Y_train = process(n_lines, args.number)
    
    df = pd.DataFrame(Y_train, columns=["id", "label"])
    df.to_csv(args.fout, index=False, sep=',')
