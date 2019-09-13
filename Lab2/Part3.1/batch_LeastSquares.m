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

abs_err_sin = [];
abs_err_sq = [];

abs_err_sin_val = [];
abs_err_sq_val = [];

for n = 2:63 % n = number of RBF


mean = linspace(0+pi/n,2*pi-pi/n,n); % Equispaced RBF (between 0 and 2pi)
sigma = (mean(end)-mean(end-1))/4; % std_dev

%Plot of RBFs

% t = 0:0.01:2*pi;
% figure(2)
% hold on
% for i = 1:length(mean)
%     plot(t,exp(-(t-mean(i)).^2/sigma.^2),'LineWidth',1.5)
% end
% axis([0 2*pi 0 1.2])
% title('RBFs')
% xlabel('x')
% ylabel('\phi_{i}(x)')
% hold off

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

end


figure(3)
hold on
plot(2:n,abs_err_sin,'r','LineWidth',1.5)
plot(2:n,abs_err_sin_val,'b','LineWidth',1.5)
title('Absolute Residual Error vs number of RBF sin(2x)')
xlabel('number of RBF')
ylabel('Absolute Residual Error')
legend('Training error','Validation Error','Location','NorthEast')
%axis([10 63 0 0.4])
hold off

figure(4)
hold on
plot(2:n,abs_err_sq,'r','LineWidth',1.5)
plot(2:n,abs_err_sq_val,'b','LineWidth',1.5)
title('Absolute Residual Error vs number of RBF square(2x)')
xlabel('number of RBF')
ylabel('Absolute Residual Error')
legend('Training error','Validation Error','Location','NorthEast')
%axis([10 63 0 0.4])
hold off




