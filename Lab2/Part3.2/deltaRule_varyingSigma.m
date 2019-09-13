%% RBF using delta rule (variation of RBFs)

clc
clear
close all

% Data generation and plot

x = (0:0.1:2*pi)' + sqrt(0.1)*randn(63,1); % input
x_val = (0.05:0.1:2*pi)' + sqrt(0.1)*randn(63,1);; % validation samples
N = length(x); % Number of input samples

f = sin(2*x); % sin(2x)
f_val = sin(2*x_val); % sin(2x) (validation)

% f = 1*(f>=0) -1 * (f<0);; % square(2x)
% f_val = *(f_val>=0) -1 * (f_val<0); % square(2x) (validation)

nMax = 15; % max number RBF

sigmas = 0.5:0.2:2;
nSigmas = length(sigmas);

val_err_nRBF = zeros(nSigmas,nMax-1);
for m = 1:nSigmas
    m
    for n = 2:nMax % number of RBF

        mean = linspace(0+pi/n,2*pi-pi/n,n); % Equispaced RBF (between 0 and 2pi)
        sigma = sigmas(m); % std_dev

        eta = 0.001;
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

        val_err_nRBF(m,n-1) = err_val(end); 

    end
end
plot(2:nMax,val_err_nRBF,'LineWidth',1.5)
title('Absolute Residual Error vs number of RBF sin(2x)')
xlabel('number of RBF')
ylabel('abs. error')
legend('\sigma = 0.5','\sigma = 0.7','\sigma = 0.9', ...
    '\sigma = 1.1','\sigma = 1.3','\sigma = 1.5','\sigma = 1.7', ...
    '\sigma = 1.9');
grid on

