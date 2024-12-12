#!/bin/bash

#directory="C:\Users\cpham\OneDrive\Documents\OBDwiz\CSV Logs\JF1ZNBE19R8754301"
directory="data/"
anaconda_python="C:/ProgramData/anaconda3/python.exe"

if [ -d "$directory" ]; then
    for file in $(ls "$directory"); do
        echo $directory/$file
        $anaconda_python  test_model.py "$directory/$file"
    done;
fi

while True ; do
    sleep 10s
done;