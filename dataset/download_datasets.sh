#!/bin/bash

# ABC Dataset downloading
FILEID=1hET43eM2cfwfqI7g1VsusU1wNpvc2VHl
FILENAME=abc.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt

# Downloading backgrounds for making synthetic dataset for cleaning
FILEID=13eJ9PSYg3QgSuTE4nXgnVVl17k6lTEb7
FILENAME=background.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt

FILEID=1dgJLgtPvk9SK9rOCnw-WQRBG6OsMUEH3
FILENAME=Dataset_of cleaning.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt


#downloading Precision floorplan from https://www.precisionfloorplan.com/floorplan-database

python3 precision_floorplan_download.py

FILEID=1dDs06LsLNQUg9HvUwNBIq-95bjmRAiMh
FILENAME=golden_set.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt



# If doesn't work try this type of function
#
#function gdrive_download () {
#  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$1" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
#  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
#  rm -f /tmp/cookies.txt
#}
#
#gdrive_download 1HIvMOJqm77flpvJWNpLHSxOpm8et_s_W Background.tar.gz
#
#
#tar -zxvf Background.tar.gz
#rm Background.tar.gz
#mv Background ../data/Background
#rm -rf __MACOSX


