%% Multilayer

clear
close all

% Data generation

nInputs = 2; % Number of inputs (xi)
nHidden = 4; % Number of hidden nodes (hj)
nOutputs = 1; % Number of outpus (yk)


% Input data

nData = 400;
nTraining = 289;
nTest = nData-nTraining;

x=linspace(-5,5,sqrt(nData))';
y=linspace(-5,5,sqrt(nData))';
z=exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
figure(1)
mesh(x, y, z);

T = reshape (z, 1, nData);
[xx, yy] = meshgrid (x, y);
X = [reshape(xx, 1, nData); reshape(yy, 1, nData); ones(1,nData)]; % inputs and the bias

Xtest = X(:,1:nTest);
X = X(:,nTest+1:nData);
T = T(nTest+1:nData);

% TRAINING
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
while deltaErr>=1e-7
    hin = W * X; % We add here the ones (bias)
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,nTraining)]; % Also add of the bias
    oin = V * hout;
    out = 2 ./ (1+exp(-oin)) - 1;
    
    delta_o = (out - T) .* ((1 + out) .* (1 - out)) * 0.5;
    delta_h = (V' * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
    delta_h = delta_h(1:nHidden, :);
    
    dw = (dw .* alpha) - (delta_h * X') .* (1-alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
    W = W + dw .* eta;
    V = V + dv .* eta;
    
    mse(end+1) = 0.5*sum(diag((out-T)'*(out-T)))/length(out-T);
    if (length(mse)>1)
        deltaErr = abs(mse(end-1) - mse(end));
    end
    Xtest = [Xtest X];

    hin_val = W * Xtest; % We add here the ones (bias)
    hout_val = [2 ./ (1+exp(-hin_val)) - 1 ; ones(1,nData)]; % Also add of the bias
    oin_val = V * hout_val;
    out_val = 2 ./ (1+exp(-oin_val)) - 1;
    
    zz = reshape(out_val, sqrt(nData), sqrt(nData));
    %mesh(x,y,zz);
    axis([-5 5 -5 5 -0.7 0.7]);
    drawnow;
    
    T_val = reshape(exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5,1,nData);
    mse_val(end+1) = 0.5*sum(diag((out_val-T_val)'*(out_val-T_val)))/length(out_val-T_val);
    Xtest = Xtest(:,1:nTest);
end

% Plot approximation
figure(2)
mesh(x,y,zz);
title('Approximated function (12 Hidden Nodes)')

% Plot evolution mse
figure(3)
hold on
plot(mse,'g','LineWidth',1.5)
plot(mse_val,'k','LineWidth',1.5)
xlabel('Epochs')
ylabel('mse')
title('Evolution of mse with number of epochs (12 Hidden Nodes)')
legend('training mse','validation mse','Location','NorthEast')
grid on
hold off










