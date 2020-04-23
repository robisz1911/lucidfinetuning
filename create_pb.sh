#!/bin/bash

for (( i = 1; i <= 20; i++ ))
do
    python create_pb.py --epoch=$i
done

for (( i = 21; i <= 40; i+=3 ))
do
    python create_pb.py --epoch=$i
done

for (( i = 45; i <= 100; i+=5 ))
do
    python create_pb.py --epoch=$i
done

for (( i = 110; i <= 200; i+=10 ))
do
    python create_pb.py --epoch=$i
done

for (( i = 250; i <= 600; i+=50 ))
do
    python create_pb.py --epoch=$i
done

