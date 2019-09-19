% Cl vs Manual RBF Means

%% Generation of CL means
clc
clear
close all

% Data generation and plot

x = (0:0.1:2*pi)'; % input
x_val = (0.05:0.1:2*pi)'; % validation samples
N = length(x); % Number of input samples

f = sin(2*x); % sin(2x)
f_val = sin(2*x_val); % sin(2x) (validation)

% x = x + sqrt(0.1)*randn(63,1); % input noise
% x_val = x_val + sqrt(0.1)*randn(63,1); % validation samples noise

% RBFs

n = 12; % Number of RBF

mean = 2*pi*rand(1,n,'double'); % n uniform distribute random numbers in [0 1]
sigma = 1.2; % Sigma that worked well in 3.2

deltaErr = 1000;
mean_initial = mean; % vector to save initial values of mean;

% Calculaiton of RBF centers
eta = 0.05;
for epoch = 1:1000 % 100 epochs enough
    index = randperm(N);
    xk = x(index); % data shuffle and normalization
    mean_prev = mean;
    % iteration in the whole dataset
    for i = 1:N
        distances = abs(mean - xk(i)); % obtain distances from x to each RBF center
        index_winner = find(distances == min(distances)); % find index of the winner
        mean(index_winner) = mean(index_winner) + eta*(xk(i) - mean(index_winner));
    end
end
hold on
plot(x,zeros(1,N),'ok')
plot(mean,zeros(1,n),'or','MarkerFaceColor','r')
title('RBF positions')
legend('Input Samples','RBF positions (mean)','Location','NorthWest')
axis([0 2*pi -1 1])
hold off

%% Data generation (no noise)

x = (0:0.1:2*pi)'; % input
x_val = (0.05:0.1:2*pi)'; % validation samples
N = length(x); % Number of input samples

f = sin(2*x); % sin(2x)
f_val = sin(2*x_val); % sin(2x) (validation)

%% Delta Rule for CL

eta = 0.001;
w = 0.2*randn(n,1);
deltaW = zeros(n,1);

err_CL = zeros(1,1000);

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
    err_CL(epoch) = err_abs;
end

%% Addition of noise

x = x + sqrt(0.1)*randn(63,1); % input + noise
x_val = x_val + sqrt(0.1)*randn(63,1); % validation samples + noise

%% Delta Rule for CL (noise)

eta = 0.001;
w = 0.2*randn(n,1);
deltaW = zeros(n,1);

err_CL_noise = zeros(1,1000);

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
    err_CL_noise(epoch) = err_abs;
end


%% Removing noise

x = (0:0.1:2*pi)'; % input
x_val = (0.05:0.1:2*pi)'; % validation samples

%% Delta Rule for manual (no noise)

mean = linspace(0+pi/n,2*pi-pi/n,n); % Equispaced RBF (between 0 and 2pi)

eta = 0.001;
w = 0.2*randn(n,1);
deltaW = zeros(n,1);

err_manual = zeros(1,1000);

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
    err_manual(epoch) = err_abs;
end


%% Addition of noise

x = x + sqrt(0.1)*randn(63,1); % input + noise
x_val = x_val + sqrt(0.1)*randn(63,1); % validation samples + noise

%% Delta Rule for manual (with noise)

mean = linspace(0+pi/n,2*pi-pi/n,n); % Equispaced RBF (between 0 and 2pi)

eta = 0.001;
w = 0.2*randn(n,1);
deltaW = zeros(n,1);

err_manual_noise = zeros(1,30000);

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
    err_manual_noise(epoch) = err_abs;
end

%% Plots
figure(2)
hold on
plot(1:length(err_CL),err_CL,'b','LineWidth',1.5)
plot(1:length(err_CL_noise),err_CL_noise,'b--','LineWidth',1.5)
plot(1:length(err_manual),err_manual,'r','LineWidth',1.5)
plot(1:length(err_manual_noise),err_manual_noise,'r--','LineWidth',1.5)
xlabel('epochs')
ylabel('Absolute Residual Error')
title('Comparison Error CL vs manual (\sigma = 1.2 and nRBF = 12)')
legend('CL clean','CL with noise','Manual clean','Manual with noise', ...
    'Location','NorthWest')
hold off