#/bin/bash


AUTODOCK_PATH=`pwd`/autodock4

cd ./EXAMPLES/1dwd/dock_rigidlig

OUTPUT="output_logfile.dlg"
INPUT="1dwd_1dwd.dpf"

echo "Running Autodock in directory: " $PWD

echo "Input: " $INPUT
echo "Output: " $OUTPUT

if [ "$1" = "nvprof" ]; then
    echo "Starting autodock with nvidia profiler..."

    nvprof $AUTODOCK_PATH -p $INPUT -l $OUTPUT
elif [ "$1" = "gdb" ]; then
    echo "Starting autodock with GDB"
    gdb --args $AUTODOCK_PATH -p $INPUT -l $OUTPUT
else
    echo "Starting autodock..."
    $AUTODOCK_PATH -p $INPUT -l $OUTPUT
fi
