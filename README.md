# AutoDockCuda

AutoDockCuda is a GPU accelerated impementation of Autodock 4.2.6, using Nvidia CUDA. It requires CUDA-capable hardware to run. It was developed by students at Portland State University as part of CS510 - Accelerated Computing with GPUs, APUs and FPGAs, under Prof. Karen Karavanic.

AutoDockCuda is not complete, and should NOT be used in research in its present state..

AutoDockCuda has not been under active develpment.

To run AutoDockCuda, you will first need to build it from source. After cloning the repository, cd into it and go to the autodock source directory:

> cd autodock-4.2.6/src/autodock

Then, run the configuration script. You must use the -enable-cuda flag and give a path to the cuda base directory:

> ./configure --enable-cuda --with-cuda=[path to CUDA toolkit directory]

Example:

> ./configure --enable-cuda --with-cuda=/pkgs/nvidia/5.5

Finally, make the project:

> make


The autodock executable is autodock4, it will be created in the same 
directory. See autodock4 documentation for information on using autodock4.

Our project comes with a crude shell script to automate running autodock. It is included in the source directory.
To run it:

> ./runit.sh <option>

<option> may be left blank. use option 'gdb' to run in gdb, or 'nvprof' to run with nvidia profiler.


