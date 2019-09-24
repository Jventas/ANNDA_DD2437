%% Excercise 3.1

clear
clc
close all

wrongAttractors = zeros(1,3);

for ii = 1:1000
% Define memory patterns

x1=[-1 -1 1 -1 1 -1 -1 1];
x2=[-1 -1 -1 -1 -1 1 -1 -1];
x3=[-1 1 1 -1 -1 1 -1 1];

X = [x1; x2; x3]; % training matrix X

% Getting W matrix
W = zeros(length(x1));

for i=1:size(X,1)
    W = W + X(i,:)'*X(i,:);
end

W = W/size(X,1);
% W = W - eye(size(W,1));

% Distorted patterns

x1d=[ 1 -1 1 -1 1 -1 -1 1]; % 1 bit error
x2d=[ 1 1 -1 -1 -1 1 -1 -1]; % 2 bit errors
x3d=[ 1 1 1 -1 1 1 -1 1]; % 2 bit errors

X_d = [x1d; x2d; x3d];
nUnchanged = 0;
X_out = X_d;

    for i=1:size(X_d,1) % iteration in all distorted patterns
        nUnchanged = 0;
        while nUnchanged < 1 % iteration: stop when it stops changing
            index = randperm(size(X_d,2));
            previous = X_out(i,:);
            for j = 1:length(index)
                X_out(i,index(j)) = sign(W(index(j),:)*X_out(i,:)');
            end
            if sum(previous - X_out(i,:)) == 0
                nUnchanged = nUnchanged + 1;
            else
                nUnchanged = 0;
            end
        end
    end
   wrongAttractors = wrongAttractors + sign(sum(not(X_out == X),2)');
end

wrongAttractors = wrongAttractors/ii % Proportion of wrong attractors obtained for each pattern
