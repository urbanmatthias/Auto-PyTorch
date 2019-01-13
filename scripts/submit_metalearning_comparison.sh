#!/bin/bash

RUN=0
CONFIG=0
INSTANCE=200
RUN_SCRIPT=run_metalearning_comparison.moab
export AUTONET_HOME=$PWD

mkdir outputs
cd outputs

while [ $INSTANCE -le 229 ]
do

  while [ $RUN -le 3 ]
  do

    while [ $CONFIG -le 3]
    do

      export INSTANCE
      export CONFIG
      export RUN
      OPDIR=output_${INSTANCE}_${CONFIG}_${RUN}
      mkdir $OPDIR
      cp $AUTONET_HOME/scripts/$RUN_SCRIPT $OPDIR
      cd $OPDIR
      export OUTPUTDIR=$PWD
      msub $RUN_SCRIPT | sed '/^\s*$/d' >> $AUTONET_HOME/jobs.txt
      cd ..
    
      let CONFIG=$CONFIG+1
    done
  
    let RUN=$RUN+1
  done 

  let INSTANCE=$INSTANCE+1
done
