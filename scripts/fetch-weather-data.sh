#!/bin/bash

months="03 04 05 06 07 08 09"

for mm in $months
do
    end=$(expr $mm + 1)
    echo $end
    wget -O ../data/weather_$mm.json \
        "http://api.worldweatheronline.com/premium/v1/\
past-weather.ashx?key=bd22daa9e95b490cbe9221301181510&\
q=New%20Delhi&format=json&date=2018-$mm-01&enddate=2018-$end-01&tp=1"

done

# most recent month
wget -O ../data/weather_10.json \
        "http://api.worldweatheronline.com/premium/v1/\
past-weather.ashx?key=bd22daa9e95b490cbe9221301181510&\
q=New%20Delhi&format=json&date=2018-10-01&enddate=2018-10-20&tp=1"