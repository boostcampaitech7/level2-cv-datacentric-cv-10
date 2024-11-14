#!/bin/bash

pip install --upgrade pip # pip 패키지 업데이트
apt update # 패키지 목록 업데이트
apt install build-essential # g++ compiler 다운로드
apt install wget # wget 다운로드
apt install make # make 다운로드

# url initialization
DATA_URL='https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000315/data/20240912160112/data.tar.gz'
CODE_URL='https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000315/data/20241024082938/code.tar.gz'
DEPENDENCY_PATH='code/requirements.txt'

# data download
wget "$DATA_URL"
tar -zxvf data.tar.gz

# code download
wget "$CODE_URL"
tar -zxvf code.tar.gz

rm -rf *.tar.gz

pip install -r "$DEPENDENCY_PATH"