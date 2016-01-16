#!/bin/bash
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Mon 07 Dec 2015 03:12:23 PM CST
#
#

python parse_log.py $1 ${1%/*}
python plot.py $1 'test' 
python plot.py $1 'train'

echo 'plot finish!'
