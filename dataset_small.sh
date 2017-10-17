#!/usr/bin/env bash

wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

unzip -o "ml-latest-small.zip"
DESTINATION="./datasets/"
mkdir -p $DESTINATION
mv ml-latest-small $DESTINATION
