disp('This test checks if two runs of AutoDock yield similar results');

num_torsions = input('Enter the number of torsions of the input ligand: ', 's');
num_torsions = str2num(num_torsions);

control_filename = input('Enter the name of the control data file: ', 's');

%   load in the control data file
control_rawdata = load (control_filename, '-ascii');

experimental_filename = input('Enter the name of the experimental data file: ', 's');

%   load in the experimental data file
experimental_rawdata = load (experimental_filename, '-ascii');

[control_e, control_p, nument1] = format_data(control_rawdata, num_torsions);
[experimental_e, experimental_p, nument2]  = format_data(experimental_rawdata, num_torsions);


[Pe,Pp] = test_two_outputs_are_similar_non_parametric(control_e, experimental_e, control_p, experimental_p, nument1);

disp('The P-value from the Wilcoxon rank sum test on the free binding energies is: ');
Pe

disp('The P-value from the Wilcoxon rank sum test on the poses is: ');
Pp
