#!/usr/bin/env bash

wget http://files.grouplens.org/datasets/movielens/ml-latest.zip

unzip -o "ml-latest.zip"
DESTINATION="./datasets/"
mkdir -p $DESTINATION
mv ml-latest $DESTINATION
