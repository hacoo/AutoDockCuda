#!/bin/sh
#hostname
PATH=/packages/autodock:${PATH}
export PATH
tar xf input1.tar
autodock4 -p test_1_parameterfile.dpf -l test_1_logfile.${1}.dlg
rm test_1_parameterfile.dpf 1adb*
exit 0
