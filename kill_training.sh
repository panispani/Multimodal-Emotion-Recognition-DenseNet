#!/bin/bash
PID=`ps -ef | grep pp3414 | grep emotion_train | grep -v grep | awk '{print $2}'`
kill -KILL $PID
PID=`ps -ef | grep pp3414 | grep emotion_eval | grep -v grep | awk '{print $2}'`
for i in $PID; do
    kill -KILL $i
done
