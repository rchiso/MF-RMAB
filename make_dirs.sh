#!/bin/bash

directory="./Exp_data"
 
if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
    echo "Directory '$directory' created."
else
    echo "Directory '$directory' already exists."
fi


directory="./Exp_data/Pulls"
 
if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
    echo "Directory '$directory' created."
else
    echo "Directory '$directory' already exists."
fi

directory="./Exp_data/Engaged"
 
if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
    echo "Directory '$directory' created."
else
    echo "Directory '$directory' already exists."
fi

directory="./Exp_data/TrueProb"
 
if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
    echo "Directory '$directory' created."
else
    echo "Directory '$directory' already exists."
fi
 
directory="./Benchmark"
 
if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
    echo "Directory '$directory' created."
else
    echo "Directory '$directory' already exists."
fi


directory="./Benchmark/Pulls"
 
if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
    echo "Directory '$directory' created."
else
    echo "Directory '$directory' already exists."
fi

directory="./Benchmark/Rewards"
 
if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
    echo "Directory '$directory' created."
else
    echo "Directory '$directory' already exists."
fi

