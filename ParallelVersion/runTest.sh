#! /bin/bash

# arg1: testVal

outfile="output.txt"
echo "size, time serial, time parallel" > "$outfile"

function runConfiguration {
    #OLDIFS="$IFS"
    #IFS="~"
    size=$1
    testVal=$2

    echo "--------- executing configuration: size=$size test value =$testVal ---------------"
    echo ""

    times=($( ./md5Gpu "$testVal" -s ))
    timep=($( ./md5Gpu "$testVal" ))

    ####### output=($( { ./md5Gpu -i "$testVal" 1> "testout.txt" ; } 2>&1 ))
    ##exitStatus=$?

    ##if (test "$exitStatus" -ne "0"); then
    ##    echo "There was a problem running the configuration: $output" 1>&2
    ##fi
    
    ##echo "$size,$times,$timep"

    ##IFS="$OLDIFS"
}

# -------------- testVal    
runConfiguration 5 "zzzzz"
runConfiguration 4 "zzzz"
runConfiguration 3 "zzz"
runConfiguration 2 "zz"
runConfiguration 5 "aaaaa"
runConfiguration 4 "aaaa"
runConfiguration 3 "aaa"
runConfiguration 2 "aa"



