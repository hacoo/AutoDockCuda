#!/bin/sh
#hostname
PATH=/packages/autodock:${PATH}
export PATH
tar xf input3.tar
autodock4 -p test_3_parameterfile.dpf -l test_3_logfile.${1}.dlg
rm test_3_parameterfile.dpf 3ptb*
exit 0
