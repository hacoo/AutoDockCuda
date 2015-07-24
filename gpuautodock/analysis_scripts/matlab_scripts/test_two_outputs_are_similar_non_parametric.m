
%   This function returns p-values from the Wilcoxon rank-sum test on the
%   hypothesis that the samples of free binding energies (Pe) and poses (Pp) from
%   two different sets of runs of AutoDock come from the same distribution
function [Pe,Pp] = test_two_outputs_are_similar_non_parametric(energies1, energies2, poses1, poses2, nument)
[m1,n1] = size(energies1);
[m2,n2] = size(energies2);
m1
m2

figure(100);
hist(energies1);
figure(101);
hist(energies2);

u = energies1;
v = energies2;
%[He,Pe] = ttest2(X,Y,0.5,'both','unequal');
[Pe,He] = ranksum(u,v);
         




%Pose testing
p1 = zeros(m1,nument);
p2 = zeros(m2,nument);

for i=1:m1
    for j=1:nument
        
    p1(i,j) = poses1((i-1)*nument + j);
   
    end
end

for i=1:m2
    for j=1:nument
        
    p2(i,j) = poses2((i-1)*nument + j);
   
    end
end

%normalize all values to [0,1] interval
p = [p1;p2];

for j=1:nument
    a = min(p(:,j));
    b = max(p(:,j));
    
    for i=1:m1
        p1(i,j) = (p1(i,j) - a)/(b-a);
    end
    for i=1:m2
        p2(i,j) = (p2(i,j) - a)/(b-a);
    end
    
end

% c is centroid pose. its a row array
c = zeros(1,nument);

%compute the centroid from control runs
for j=1:nument
    c(j) = mean(p1(:,j));
end

%compute each pose's distance from the centroid
p1d = zeros(m1,1);
p2d = zeros(m2,1);

for i=1:m1

    d = sum(((p1(i,:)-c).^2))^0.5;
    p1d(i) = d;
end

for i=1:m2

    d = sum(((p2(i,:)-c).^2))^0.5;
    p2d(i) = d;
end

min_energies1 = min(energies1)
min_energies2 = min(energies2)
max_energies1 = max(energies1)
max_energies2 = max(energies2)


figure(200);
hist(p1d,10);
figure(201);
hist(p2d,10);
[Pp,Hp] = ranksum(p1d,p2d);
%[Hp,Pp] = ttest2(p1d,p2d,0.5,'both','unequal');
        
end %function

























