%% Batch mode using Least Squares

clear
close all


% Data generation and plot

x = (0:0.1:2*pi)'; % input
x_val = (0.05:0.1:2*pi)'; % validation samples
N = length(x); % Number of input samples

f_sin = sin(2*x); % sin(2x) (target 1)
f_sq = 1*(f_sin>=0) -1 * (f_sin<0); % square(2x) (target 2)

f_sin_val = sin(2*x_val); % sin(2x) (validation)
f_sq_val = 1*(f_sin_val>=0) -1 * (f_sin_val<0); % square(2x) (validation)
% Plot of both sin(2x) and square(2x)

figure(1)
hold on
plot(x,f_sin,'r','LineWidth',1.5)
plot(x,f_sq,'b','LineWidth',1.5)
legend('sin(2x)','square(2x)','Location','NorthWest')
title('Target functions')
xlabel('x')
ylabel('f(x)')
axis([0 2*pi -1.5 1.5])
grid on
hold off

% RBF generation (mean and variance)

n = 12; % Number of RBF

abs_err_sin = [];
abs_err_sq = [];

abs_err_sin_val = [];
abs_err_sq_val = [];




mean = linspace(0+pi/n,2*pi-pi/n,n); % Equispaced RBF (between 0 and 2pi)
sigma = 1.2; % std_dev

%Plot of RBFs

t = 0:0.01:2*pi;
figure(2)
hold on
for i = 1:length(mean)
    plot(t,exp(-0.5*(t-mean(i)).^2/sigma.^2),'LineWidth',1.5)
end
axis([0 2*pi 0 1.2])
title('RBFs')
xlabel('x')
ylabel('\phi_{i}(x)')
hold off

phi = zeros(N,n);

% Populate phi matrix
for k=1:n
    phi(:,k) = exp(-0.5*(x-mean(k)).^2/sigma.^2);
end

w_sine = inv(phi'*phi)*phi'*f_sin;
w_sq = inv(phi'*phi)*phi'*f_sq;

% Validation and Error calculation (using hold-out validation set)

% Populate phi_val matrix

phi_val = zeros(N,n);

for k=1:n
    phi_val(:,k) = exp(-0.5*(x_val-mean(k)).^2/sigma.^2);
end

abs_err_sin(end+1) = sum(abs(phi*w_sine-f_sin))/length(abs(phi*w_sine-f_sin));
abs_err_sq(end+1) = sum(abs(phi*w_sq-f_sq))/length(abs(phi*w_sq-f_sq));

abs_err_sin_val(end+1) = sum(abs(phi_val*w_sine-f_sin_val))/length(abs(phi_val*w_sine-f_sin_val));
abs_err_sq_val(end+1) = sum(abs(phi_val*w_sq-f_sq_val))/length(abs(phi_val*w_sq-f_sq_val));

% Plot Target vs Approx

x_plot = 0:0.05:2*pi;

phi_plot = zeros(length(x_plot),n);

for k=1:n
    phi_plot(:,k) = exp(-0.5*(x_plot-mean(k)).^2/sigma.^2);
end

f_sin_plot = sin(2*x_plot);
f_sin_approx = phi_plot*w_sine;

f_sq_plot = 1*(f_sin_plot>=0) -1 * (f_sin_plot<0);
f_sq_approx = phi_plot*w_sq;

figure(3)
hold on
plot(x_plot,f_sin_plot,'r','LineWidth',1.5)
plot(x_plot,f_sin_approx,'b','LineWidth',1.5)
legend('sin(2x)','approx sin(2x)','Location','NorthWest')
title('Target vs approx')
xlabel('x')
ylabel('f(x)')
axis([0 2*pi -1.5 1.5])
grid on
hold off

figure(4)
hold on
plot(x_plot,f_sq_plot,'r','LineWidth',1.5)
plot(x_plot,f_sq_approx,'b','LineWidth',1.5)
legend('square(2x)','approx square(2x)','Location','NorthWest')
title('Target vs approx')
xlabel('x')
ylabel('f(x)')
axis([0 2*pi -1.5 1.5])
grid on
hold off





