#!/bin/bash

RUN=0
CONFIG=0
INSTANCE=0
RUN_SCRIPT=run_collect_metalearning_data.moab
export AUTONET_HOME=$PWD

mkdir outputs
cd outputs
while [ $INSTANCE -le 499 ]
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
  let INSTANCE=$INSTANCE+1
done
