#!/bin/bash

#####################################################################
#
# Use this script to pull kaiterra data from the Kaiterra server
# database. Database name: KaiterraFieldEggs
#
# IMPORTANT NOTE - The timestamps in the server are in UTC!
#
# Author: Shiva R. Iyer
#
# Date: Sep 1, 2018
#####################################################################


db_name="KaiterraFieldEggs"
tab_name="KaiterraData"
outfile_name="kaiterra_fieldeggs_all.csv"

echo "Database name: " $db_name
echo "Table name:" $tab_name
echo "Output file name:" $outfile_name
echo -n "Continue? [y/n] > "
read response
if [ "$response" != "y" -a "$response" != "Y" ]; then
    echo "Not a valid response. Expected 'y' or 'Y'."
    echo "Exiting."
    exit 2
fi

if [ -a $outfile_name ]; then
    echo -n "File $outfile_name already exists. Overwrite? [y/n] > "
    read response
    if [ "$response" != "y" -a "$response" != "Y" ]; then
	echo -n "Provide different file name: "
	read response
	if [ "$response" == "" ]; then
	    echo "Aborting because no answer was given."
	    exit 3
	else
	    outfile_name=$response
	fi
    fi
fi


# first check if count > 0
echo "Checking if table has valid entries..."
mysql_cmd="SELECT count(*) from $db_name.$tab_name"

# output=`mysql -h 18.218.35.187 -u iotuser --password=321resutoi -e "$mysql_cmd" -N`
output=`mysql -u root -p -e "$mysql_cmd" -N`

if [ $output == "0" ]; then
    echo "No results!"
else
    echo "Pulling the data..."
    echo "(IMPORTANT NOTE: Timestamps are in UTC.)"
    mysql_cmd="SELECT 'time','device_udid','short_id','pm_25','pm_10' UNION ALL SELECT time,device_udid,short_id,pm_25,pm_10 FROM $db_name.$tab_name ORDER BY time ASC"
    
    # mysql -h 18.218.35.187 -u iotuser --password=321resutoi -e "$mysql_cmd" -N | sed 's/\t/,/g' > $outfile_name
    mysql -u root -p -e "$mysql_cmd" -N | sed 's/\t/,/g' > $outfile_name
    echo "Exported to $outfile_name. Done."
fi
    
unset mysql_cmd
