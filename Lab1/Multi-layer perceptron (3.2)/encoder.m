%% Multilayer

clear
close all
% Data generation

nInputs = 8; % Number of inputs (xi)
nHidden = 3; % Number of hidden nodes (hj)
nOutputs = 8; % Number of outpus (yk)


% Input data


X = 2*eye(nInputs) - 1;

T = X;
X = [X;ones(1,nInputs)];

W = 0.2*randn(nHidden,nInputs+1); % Random initialization of W (dimension nHidden x nInputs+1)
V = 0.2*randn(nOutputs,nHidden+1); % Random initialization of V (dimension nOutputs x nHidden+1)

hin = []; 
hout = []; 
oin = [];
out = [];

mse = [];
deltaErr = 1000;
alpha = 0.9; % alpha used in the last step of the algorithm
eta = 2; % learning rate

dw = 0; % initial value for deltaW
dv = 0; % initial value for deltaV

while deltaErr>=1e-7
    hin = W * X; % We add here the ones (bias)
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,nInputs)]; % Also add of the bias
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
    mse(end);
end

% Plot evolution mse
% figure
% plot(mse,'o-g','MarkerFaceColor','g')
% xlabel('epoch')
% ylabel('mse')
% title('Evolution of mse along epochs')
% grid on

% Plot evolution mse
figure
plot(mse,'o-g','MarkerFaceColor','g')
xlabel('Epochs')
ylabel('mse')
title('Evolution of mse with number of epochs')
grid on




