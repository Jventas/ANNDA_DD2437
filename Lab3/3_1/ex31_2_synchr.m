%% Excercise 3.1_2

clear
close all

d = 8; % number of neurons 

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

% Calculation of atractors

attractors = zeros(2^d,d);

for i = 1:2^d
    x = 2*de2bi(i-1,d,'left-msb')-1; % obtain input i
    % Obtain atractor i
    x_out = x;
    for k=1:100
        x_out = Sign((W*x_out'))';
    end
    attractors(i,:) = x_out;
end

attractors = unique(attractors,'rows');



