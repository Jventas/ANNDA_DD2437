%% RBF using delta rule (variation of RBFs)

clc
clear
close all

% Data generation and plot

x = (0:0.1:2*pi)'; % input
x_val = (0.05:0.1:2*pi)'; % validation samples
N = length(x); % Number of input samples

f = sin(2*x); % sin(2x)
f_val = sin(2*x_val); % sin(2x) (validation)

% f = 1*(f>=0) -1 * (f<0);; % square(2x)
% f_val = 1*(f_val>=0) -1 * (f_val<0); % square(2x) (validation)

x = x + sqrt(0.1)*randn(63,1); % input
x_val = x_val + sqrt(0.1)*randn(63,1); % validation samples

nMax = 16; % max number RBF

val_err_nRBF = zeros(2,nMax-1);

% Batch RBFs
for n = 2:nMax % number of RBF
    n
    mean = linspace(0+pi/n,2*pi-pi/n,n); % Equispaced RBF (between 0 and 2pi)
    sigma = 1.2; % std_dev
    
    %Batch mode
    phi_batch = Gaussian(x,mean,sigma);
    w_batch = inv(phi_batch'*phi_batch)*phi_batch'*f;
    
    phi_val = Gaussian(x_val,mean,sigma);
    val_err_nRBF(1,n-1) = sum(abs(phi_val*w_batch-f_val))/length(abs(phi_val*w_batch-f_val));

end

X = [x'; ones(1,length(x))]; % inputs and the bias

Xtest = [x_val'; ones(1,length(x_val))];
T = f';
Ttest = f_val';

% MLP
for n = 2:nMax % number of Hidden nodes
    n
    % TRAINING
    nInputs = 1; % Number of inputs (xi)
    nHidden = n; % Number of hidden nodes (hj)
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

    for epochs = 1:30000
        hin = W * X; % We add here the ones (bias)
        hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,length(hin))]; % Also add of the bias
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

        hin_val = W * Xtest; % We add here the ones (bias)
        hout_val = [2 ./ (1+exp(-hin_val)) - 1 ; ones(1,length(hin_val))]; % Also add of the bias
        oin_val = V * hout_val;
        out_val = 2 ./ (1+exp(-oin_val)) - 1;
    end
    val_err_nRBF(2,n-1) = sqrt(mse(end)); 
end

plot(2:n,val_err_nRBF,'LineWidth',1.5)
title('Absolute Residual Error')
xlabel('number of RBF')
ylabel('abs. error')
legend('RBF','MLP','Location','NorthWest');
grid on
axis([2 nMax 0 max(max(val_err_nRBF)) + 0.05])

