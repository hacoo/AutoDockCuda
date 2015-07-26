function [e, p, nument] = format_data(rawdata, numtorsions)

length = size(rawdata);

%   This variable is a relic of an older version of this script.
%   We set it to 1 because we assume that we want to treat all data
%   as coming from the same input 
numexperiments = 1;
numruns = rawdata(length(1));

%   numentries is the number of torsions plus
%   3 for x,y,z plus 
%   4 for orientation plus 
%   and 1 for the final binding energy of that pose
numentries = 3 + 4 + 1 + numtorsions; 
numrows = numentries*numruns;

%   data will be configured as a matrix of values. Each column represnts an
%   experiment. There are 'numexperiments' number of experiments. 
%   Each experiment consists of 'numruns' number of runs of the
%   genetic algorithm. Each run has 'numentries' data values associated with
%   it. Each element of each column represents one of these data values. In
%   order they are
%   (1 - 3): X,Y,Z position of the pose
%   (4 - 7): Orientation of the pose
%   (8 - (numtorsions + 7)): angle measures for each torsion
%   (numentries): the final value will be the binding energy for that pose




%   set up initial data array %
data = zeros(numrows, numexperiments);

for j=1:numexperiments
    for i=1:numrows
        data(i,j) = rawdata((j-1)*numrows + i);
    end %for i
end %for j

%   set up energy array %
energies = zeros(numruns, numexperiments);

for j=1:numexperiments
    for i=1:numruns
        energies(i,j) = data(i*numentries, j);
    end %for i
end %for j

%   set up poses array %
numrows_poses = (numentries-1)*numruns;
poses = zeros(numrows_poses, numexperiments);
counter = 0;

for j=1:numexperiments
    for i=1:numrows
        k = i;
        if (mod(k,numentries) == 0)
            counter = counter +1;
        else
            poses(i-counter,j) = data(i,j);
        end
        
    end %for i
    
end %for j

%return values of this function
e = energies;
p = poses;
nument = (3 + 4 + numtorsions);



end %function

























