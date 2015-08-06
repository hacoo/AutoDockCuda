#/bin/bash

cd /u/hcooney/CS510GPGPU/AutoDockCuda/autodock-4.2.6-src/src/autodock/EXAMPLES/1dwd/dock_rigidlig

OUTPUT="output_logfile.dlg"
INPUT="1dwd_1dwd.dpf"
AUTODOCK_PATH="/u/hcooney/CS510GPGPU/AutoDockCuda/autodock-4.2.6-src/src/autodock/autodock4"

echo "Running Autodock in directory: " $PWD

echo "Input: " $INPUT
echo "Output: " $OUTPUT

echo "Starting autodock with nvidia profiler..."

nvprof $AUTODOCK_PATH -p $INPUT -l $OUTPUT

