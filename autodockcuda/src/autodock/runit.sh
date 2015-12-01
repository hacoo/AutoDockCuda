#/bin/bash


AUTODOCK_PATH=`pwd`/autodock4


cd ./EXAMPLES/1dwd/dock_flexlig
egdb() { emacs --eval "(gdb \"gdb -cd ~/CS510GPGPU/AutoDockCuda/autodock-4.2.6-src/src/autodock/EXAMPLES/1dwd/dock_rigidlig -i=mi --args $*\")";}

OUTPUT="output_logfile.dlg"
INPUT="1dwd_1dwd.dpf"

echo "Running Autodock in directory: " $PWD

echo "Input: " $INPUT
echo "Output: " $OUTPUT

if [ "$1" = "nvprof" ]; then
    echo "Starting autodock with nvidia profiler..."
    nvprof $AUTODOCK_PATH -p $INPUT -l $OUTPUT
elif [ "$1" = "memcheck" ]; then
    echo "Starting autodock with nvidia profiler..."
    cuda-memcheck $AUTODOCK_PATH -p $INPUT -l $OUTPUT
elif [ "$1" = "gdb" ]; then
    echo "Starting autodock with GDB"
    gdb --args $AUTODOCK_PATH -p $INPUT -l $OUTPUT
elif [ "$1" = "egdb" ]; then
    echo "Starting autodock with GDB in EMACS (need special bashrc)"
    egdb $AUTODOCK_PATH -p $INPUT -l $OUTPUT
else
    echo "Starting autodock..."
    $AUTODOCK_PATH -p $INPUT -l $OUTPUT
fi
