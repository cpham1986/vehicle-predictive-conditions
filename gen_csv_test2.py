import os
import pandas as pd
import time

try:
    with open('data\\CSVLog_20241206_075200.csv') as f_in:
        lines = f_in.readlines()
        
    for line in lines:
        print(line)
        with open("data\\gen_test_2.csv", 'a+') as f_out:
            f_out.write(line)
        time.sleep(0.1)
    
except Exception as e:
    print(e)