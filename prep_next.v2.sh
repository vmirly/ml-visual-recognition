#!/bin/bash

temp=$1
rep=$2
inext=$3

if [ $inext -eq 0 ]; then

inext=`echo $inext | awk '{printf "%d", $1+1}'`


if [ $inext -le 20 ]; then
sed 's/__REP__/'$rep'/g' template_job.$temp.qsub | sed 's/__NUM__/'$inext'/g' >  jobs/$temp/job.rep$rep.i$inext.sh

qsub jobs/$temp/job.rep$rep.i$inext.sh
fi
fi


if [ -f results/$temp/pred.rep_$rep.$inext ]; then

inext=`echo $inext | awk '{printf "%d", $1+1}'`


if [ $inext -le 20 ]; then
sed 's/__REP__/'$rep'/g' template_job.$temp.qsub | sed 's/__NUM__/'$inext'/g' >  jobs/$temp/job.rep$rep.i$inext.sh

qsub jobs/$temp/job.rep$rep.i$inext.sh
fi
fi
