#!/bin/bash
dir=/r/output
tmp=/r/Temp
mkdir $dir

while IFS='' read -r user || [[ -n "$user" ]]; do
	user_name=`basename $user .jpg`
	echo $user_name
	./bin/imageEnhanceTest.exe  $user  $tmp/$user_name"_msr.jpg" $tmp/$user_name"_hdr.jpg" $tmp/$user_name"_altm.jpg" 1
	convert  $user  $tmp/$user_name"_msr.jpg"  $tmp/$user_name"_hdr.jpg"  $tmp/$user_name"_altm.jpg"  +append  $dir/$user_name".jpg"
done < "/r/list.txt"
