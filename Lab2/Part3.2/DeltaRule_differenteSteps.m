%% RBF using delta rule

clc
clear
close all

% Data generation and plot

x = (0:0.1:2*pi)'; % input
x_val = (0.05:0.1:2*pi)'; % validation samples
N = length(x); % Number of input samples

f = sin(2*x); % sin(2x)
f_val = sin(2*x_val); % sin(2x) (validation)

x = x + sqrt(0.1)*randn(63,1); % input + noise
x_val = x_val + sqrt(0.1)*randn(63,1); % validation samples + noise

n = 10; % number of RBF
mean = linspace(0+pi/n,2*pi-pi/n,n); % Equispaced RBF (between 0 and 2pi)
sigma = (mean(end)-mean(end-1))/4; % std_dev

etas = [0.05 0.01 0.009 0.005 0.001];

err_val = zeros(5,1000);
err_tr = zeros(5,1000);

for k = 1:length(etas)

    eta = etas(k);
    w = 0.2*randn(n,1);
    deltaW = zeros(n,1);

    for epoch = 1:1000
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
        err_val(k,epoch) = err_abs;
        err_tr(k,epoch) = err;
    end
end

plot(1:length(err_val),err_val,'LineWidth',1.2)
legend('\eta = 0.05','\eta = 0.01','\eta = 0.009','\eta = 0.005','\eta = 0.001')
grid on
xlabel('epoch')
ylabel('Absolute Residual Error')
title('Influence of learning rate (\eta) in Delta Rule')
