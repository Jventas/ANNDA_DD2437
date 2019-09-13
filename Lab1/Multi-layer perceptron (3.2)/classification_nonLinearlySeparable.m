%% Multilayer
clear
close all

mse_final = zeros(1,10);
nMissclass_final = zeros(1,10);
maxHidden = 8;

for k = 8:maxHidden
k
clf('reset')
% Data generation

n = 100; % Number of data of each class
ntot = 2*n; % number of total data

mA = [ 1.0, 0.3]; sigmaA = 0.2;
mB = [ 0.0, -0.3]; sigmaB = 0.28;
classA(1,:) = [ randn(1,round(0.5*n)) .* sigmaA - mA(1), ...
randn(1,round(0.5*n)) .* sigmaA + mA(1)];
classA(2,:) = randn(1,n) .* sigmaA + mA(2);
classB(1,:) = randn(1,n) .* sigmaB + mB(1);
classB(2,:) = randn(1,n) .* sigmaB + mB(2);

p = randperm(n);

% Input data

X = [classA classB];
index = randperm(2*n);

X(1,:) = X(1,index); % X => input matrix
X(2,:) = X(2,index);
X(3,:) = ones(1,ntot); % bias

T = sign(n-index); % T => target matrix
T(T==0) = 1;

% plot data
hold on
plot(classA(1,:),classA(2,:),'or')
plot(classB(1,:),classB(2,:),'*b')
xlabel('x_{1}')
ylabel('x_{2}')
title('Data')
legend('Class A','Class B','Location','NorthWest')
axis ([min(X(1,:))-0.5 max(X(1,:))+0.5 min(X(2,:))-0.5 max(X(2,:))+0.5])
hold off
drawnow

nInputs = 2; % Number of inputs (xi)
nHidden = k; % Number of hidden nodes (hj)
nOutputs = 1; % Number of outpus (yk)

W = 0.2*randn(nHidden,nInputs+1); % Random initialization of W (dimension nHidden x nInputs+1)
V = 0.2*randn(nOutputs,nHidden+1); % Random initialization of V (dimension nOutputs x nHidden+1)

hin = []; 
hout = []; 
oin = [];
out = [];

mse = [];
deltaErr = 1000;
alpha = 0.9; % alpha used in the last step of the algorithm
eta = 0.001; % learning rate

dw = 0; % initial value for deltaW
dv = 0; % initial value for deltaV

while deltaErr>=1e-7
    hin = W * X; % We add here the ones (bias)
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ntot)]; % Also add of the bias
    oin = V * hout;
    out = 2 ./ (1+exp(-oin)) - 1;
    
    delta_o = (out - T) .* ((1 + out) .* (1 - out)) * 0.5;
    delta_h = (V' * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
    delta_h = delta_h(1:nHidden, :);
    
    dw = (dw .* alpha) - (delta_h * X') .* (1-alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
    W = W + dw .* eta;
    V = V + dv .* eta;
    
    mse(end+1) = 0.5*sum((out-T).*(out-T))/length(out-T);
    if (length(mse)>1)
        deltaErr = abs(mse(end-1) - mse(end));
    end
    mse(end);
end
mse_final(k) = mse(end);

% Calculation and plot of misclassification
decision = sign(out);
nMissclass_final(k) = sum(0.5*abs(decision-T));
end

% Plot evolution mse
% figure
% plot(mse,'o-g','MarkerFaceColor','g')
% xlabel('epoch')
% ylabel('mse')
% title('Evolution of mse along epochs')
% grid on

boundary = abs(out)<=0.8;
points_boundary = X(1:2,boundary)
hold on
plot(points_boundary(1,:),points_boundary(2,:),'ok','MarkerFaceColor','k')
legend('Class A','Class B','points in the boundary','Location','NorthWest')
hold off

% Plot evolution mse
figure
plot(mse_final,'o-g','MarkerFaceColor','g')
xlabel('Hidden nodes')
ylabel('mse')
title('Evolution of mse with number of hidden nodes')
grid on

% Plot misclassifications
figure
plot(nMissclass_final,'o-r','MarkerFaceColor','r')
xlabel('Hidden nodes')
ylabel('misclassifications')
title('Evolution of misclassifications with number of hidden nodes')
grid on


