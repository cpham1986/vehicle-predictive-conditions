#!/bin/bash

directory="C:\Users\cpham\OneDrive\Documents\OBDwiz\CSV Logs\JF1ZNBE19R8754301"

if [ -d "$directory" ]; then
    for file in $(ls "$directory"); do
        echo $directory/$file
        C:/Users/cpham/anaconda3/python.exe  test_model.py "$directory/$file"
    done;
fi