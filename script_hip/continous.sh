#!/bin/bash

for(( ; ; ))
do
	rocm-smi --csv -P -u --showmemuse 
	sleep 1
done 
