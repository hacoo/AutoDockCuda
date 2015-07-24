#!/bin/sh
#hostname
PATH=/packages/autodock:${PATH}
export PATH
tar xf input2.tar
autodock4 -p test_2_parameterfile.dpf -l test_2_logfile.${1}.dlg
rm test_2_parameterfile.dpf 7adp*
exit 0
