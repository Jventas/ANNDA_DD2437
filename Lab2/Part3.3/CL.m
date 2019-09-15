%% Competitive Learning

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
final_means = zeros(100,n);
for t = 1:100
    for epoch = 1:100 % 100 epochs enough
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
    final_means(t,:) = sort(mean);
end
hold on
plot(x,zeros(1,N),'ok')
plot(sum(final_means,1)/100,zeros(1,n),'or','MarkerFaceColor','r')
title('RBF positions')
legend('Input Samples','RBF positions (mean)','Location','NorthWest')
axis([0 2*pi -1 1])
hold off

m = sum(final_means,1)/100
var = sum((final_means-m).^2,1)/(100)


