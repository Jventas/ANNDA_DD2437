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

val_err_nRBF = zeros(4,nMax-1);

% Manual position
for n = 2:nMax % number of RBF
    n
    mean = linspace(0+pi/n,2*pi-pi/n,n); % Equispaced RBF (between 0 and 2pi)
    sigma = 1.2; % std_dev

    eta = 0.005;
    w = 0.2*randn(n,1);
    deltaW = zeros(n,1);

    err_val = zeros(1,1000);
    err_tr = zeros(1,1000);

    for epoch = 1:30000
        index = randperm(length(x));
        x_sf = x(index);
        f_sf = f(index);
        for i=1:length(x_sf)
            phi = Gaussian(x_sf(i),mean,sigma);
            err = f_sf(i) - phi*w;
            deltaW = (eta*err*phi)';
            w = w + deltaW;
        end
        % Validation
        phi_val = Gaussian(x_val,mean,sigma);
        err_abs = sum(abs(f_val - phi_val*w))/length(abs(f_val - phi_val*w));
        err_val(epoch) = err_abs;
        err_tr(epoch) = err;
    end
    
    %Batch mode
    phi_batch = Gaussian(x,mean,sigma);
    w_batch = inv(phi_batch'*phi_batch)*phi_batch'*f;
    
    phi_val = Gaussian(x_val,mean,sigma);
    val_err_nRBF(1,n-1) = err_val(end); 
    val_err_nRBF(2,n-1) = sum(abs(phi_val*w_batch-f_val))/length(abs(phi_val*w_batch-f_val));

end

% Random position
for n = 2:nMax % number of RBF
    n
    mean = 2*pi*rand(1,n); % Uniform random means
    sigma = 1.2; % std_dev

    eta = 0.005;
    w = 0.2*randn(n,1);
    deltaW = zeros(n,1);

    err_val = zeros(1,1000);
    err_tr = zeros(1,1000);

    for epoch = 1:30000
        index = randperm(length(x));
        x_sf = x(index);
        f_sf = f(index);
        for i=1:length(x_sf)
            phi = Gaussian(x_sf(i),mean,sigma);
            err = f_sf(i) - phi*w;
            deltaW = (eta*err*phi)';
            w = w + deltaW;
        end
        % Validation
        phi_val = Gaussian(x_val,mean,sigma);
        err_abs = sum(abs(f_val - phi_val*w))/length(abs(f_val - phi_val*w));
        err_val(epoch) = err_abs;
        err_tr(epoch) = err;
    end
    
    %Batch mode
    phi_batch = Gaussian(x,mean,sigma);
    w_batch = inv(phi_batch'*phi_batch)*phi_batch'*f;
    
    phi_val = Gaussian(x_val,mean,sigma);
    val_err_nRBF(3,n-1) = err_val(end); 
    val_err_nRBF(4,n-1) = sum(abs(phi_val*w_batch-f_val))/length(abs(phi_val*w_batch-f_val));

end

plot(2:n,val_err_nRBF,'LineWidth',1.5)
title('Absolute Residual Error vs number of RBF sin(2x) - \sigma = 1.2 (with noise)')
xlabel('number of RBF')
ylabel('abs. error')
legend('Delta Rule (fixed means)','LS (fixed means)', ...
    'Delta Rule (random means)','LS (random means)','Location','NorthWest');
grid on
axis([2 nMax 0 max(max(val_err_nRBF)) + 0.05])

