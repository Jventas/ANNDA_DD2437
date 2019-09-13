%% Multilayer
clear
close all

strcol1 = 'rgb';
strcol2 = '--r--g--b';
strleg = ["6 nodes training", "6 nodes val","8 nodes training", "8 nodes val","10 nodes training", "10 nodes val"];

% Data generation

n = 100; % Number of data of each class
ntot = 2*n; % number of total data
nval = 0.1*ntot;

mA = [ 1.0, 0.3]; sigmaA = 0.2;
mB = [ 0.0, -0.4]; sigmaB = 0.3;
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

Xval = X(:,1:nval);
Tval = T(:,1:nval);
X = X(:,nval+1:ntot);
T = T(:,nval+1:ntot);

for nHidden = 6:2:10

% plot data
% figure(1)
% hold on
% plot(classA(1,:),classA(2,:),'or')
% plot(classB(1,:),classB(2,:),'*b')
% xlabel('x_{1}')
% ylabel('x_{2}')
% title('Training Data')
% legend('Class A','Class B','Location','NorthWest')
% axis ([min(X(1,:))-0.5 max(X(1,:))+0.5 min(X(2,:))-0.5 max(X(2,:))+0.5])
% hold off
% drawnow

nMissclass_final = zeros(1,10);

nInputs = 2; % Number of inputs (xi)
% Hidden = 10; % Number of hidden nodes (hj)
nOutputs = 1; % Number of outpus (yk)

W = 0.2*randn(nHidden,nInputs+1); % Random initialization of W (dimension nHidden x nInputs+1)
V = 0.2*randn(nOutputs,nHidden+1); % Random initialization of V (dimension nOutputs x nHidden+1)

hin = []; 
hout = []; 
oin = [];
out = [];

mse = [];
mse_val = [];
deltaErr = 1000;
alpha = 0.9; % alpha used in the last step of the algorithm
eta = 0.001; % learning rate

dw = 0; % initial value for deltaW
dv = 0; % initial value for deltaV

for epochs = 1:35000
    hin = W * X; % We add here the ones (bias)
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ntot-nval)]; % Also add of the bias
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
    hin_val = W * Xval; % We add here the ones (bias)
    hout_val = [2 ./ (1+exp(-hin_val)) - 1 ; ones(1,nval)]; % Also add of the bias
    oin_val = V * hout_val;
    out_val = 2 ./ (1+exp(-oin_val)) - 1;
    
    mse_val(end+1) = 0.5*sum((out_val-Tval).*(out_val-Tval))/length(out_val-Tval);
    mse(end);
end


figure(2)
k = nHidden/2-2;
hold on
plot(mse,strcol1(k),'LineWidth', 1.5)
plot(mse_val,strcol2((3*(k-1)+1):(3*(k-1)+3)),'LineWidth', 1.5)
xlabel('epoch')
ylabel('mse')
title('Evolution of mse along epochs')

grid on
hold off

end
legend(strleg(1),strleg(2),strleg(3),strleg(4),strleg(5),strleg(6),'Location','NorthWest')
% boundary = abs(out)<=0.8;
% points_boundary = X(1:2,boundary);
% figure(1)
% hold on
% plot(points_boundary(1,:),points_boundary(2,:),'ok','MarkerFaceColor','k')
% legend('Class A','Class B','points in the boundary','Location','NorthWest')
% hold off


