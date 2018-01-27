#!/bin/bash
# cleanup
rm -rfv ckpt/train/*
rm -fv ckpt/log/*

# fire new background processes
nohup python emotion_eval.py &> eval_train.out &
nohup python emotion_eval.py &> eval_valid.out &
nohup python emotion_train.py & # uses nohup.out
