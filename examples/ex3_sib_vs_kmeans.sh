#!/bin/bash

script_base_name = "ex3_sib_vs_kmeans"

sleep 10

python "$script_base_name.py" prepare

sleep 10

python "$script_base_name.py" cluster

python "$script_base_name.py" evaluate
