%% Multilayer
clear
close all


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

% plot data
figure(1)
hold on
plot(classA(1,:),classA(2,:),'or')
plot(classB(1,:),classB(2,:),'*b')
xlabel('x_{1}')
ylabel('x_{2}')
title('Training Data')
legend('Class A','Class B','Location','NorthWest')
axis ([min(X(1,:))-0.5 max(X(1,:))+0.5 min(X(2,:))-0.5 max(X(2,:))+0.5])
hold off
drawnow

nMissclass_final = zeros(1,10);

nInputs = 2; % Number of inputs (xi)
nHidden = 6; % Number of hidden nodes (hj)
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

while deltaErr >= 1e-8
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

mse_batch = mse;
mse_val_batch = mse_val;

mse = [];
mse_val = [];
deltaErr = 1000;
alpha = 0.9; % alpha used in the last step of the algorithm
eta = 0.0001; % learning rate
W = 0.2*randn(nHidden,nInputs+1); % Random initialization of W (dimension nHidden x nInputs+1)
V = 0.2*randn(nOutputs,nHidden+1); % Random initialization of V (dimension nOutputs x nHidden+1)
dw = 0; % initial value for deltaW
dv = 0; % initial value for deltaV

while deltaErr >= 1e-8
    index = randperm(length(X(1,:)));
    X = X(:,index);
    T = T(index);
    e = zeros(1,length(T));
    for i = 1:length(X(1,:))
        x = X(:,i);
        t = T(i);
        hin = W * x; % We add here the ones (bias)
        hout = [2 ./ (1+exp(-hin)) - 1 ; 1]; % Also add of the bias
        oin = V * hout;
        out = 2 ./ (1+exp(-oin)) - 1;

        delta_o = (out - t) .* ((1 + out) .* (1 - out)) * 0.5;
        delta_h = (V' * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
        delta_h = delta_h(1:nHidden, :);

        dw = (dw .* alpha) - (delta_h * x') .* (1-alpha);
        dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
        W = W + dw .* eta;
        V = V + dv .* eta;
        e(i) = out-t;
    end
    
    mse(end+1) = 0.5*sum(e(end).*e(end));
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

mse_seq = mse;
mse_val_seq = mse_val;

figure(2)
k = nHidden/2;
hold on
plot(mse_batch,'k','LineWidth', 2)
plot(mse_val_batch,'r','LineWidth', 1.5)
xlabel('epoch')
ylabel('mse')
legend('mse training batch','mse validation batch','Location','NorthWest')
title('Evolution of mse along epochs (batch)')
grid on
hold off

figure(3)
k = nHidden/2;
hold on
plot(mse_seq,'k','LineWidth', 2)
plot(mse_val_seq,'r','LineWidth', 1.5)
xlabel('epoch')
ylabel('mse')
legend('mse training seq','mse validation seq','Location','NorthWest')
title('Evolution of mse along epochs (sequential)')
grid on
hold off



