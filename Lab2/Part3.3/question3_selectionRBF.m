% CL in RBF R2 -> R2

%% RBF Positions CL
clear
clc
close all

% Reading data from files

trainingData = dlmread('ballist.dat');
X = trainingData(:,1:2); % inputs for training
T = trainingData(:,3:4); % targets for training
N = length(X(:,1)); % Number of input samples

validationData = dlmread('balltest.dat');
X_val = validationData(:,1:2); % inputs for training
T_val = validationData(:,3:4); % targets for training

% RBFs

nMin = 2; % Min RBF
nMax = 40; % Max RBF

avg_dist_tr = zeros(1,nMax-nMin+1);
avg_dist_val = zeros(1,nMax-nMin+1);

for n = nMin:nMax % Number of RBF nodes


mean = rand(2,n,'double'); % n uniform distribute random numbers in [0 1]
sigma = 1.2; % Sigma

deltaErr = 1000;
mean_initial = mean; % vector to save initial values of mean;

% Calculaiton of RBF centers
eta = 0.05;
for epoch = 1:100 % 100 epochs enough
    index = randperm(N);
    Xk = X(index,1:2); % data shuffle and normalization
    mean_prev = mean;
    % iteration in the whole dataset
    for i = 1:N
        Xk_i = [Xk(i,1);Xk(i,2)];
        distances = zeros(1,n); % obtain distances from x to each RBF center
        for j = 1:n
            distances(j) = sqrt(sum((Xk_i-mean(:,j)).^2));
        end
        index_winner = find(distances == min(distances)); % find index of the winner
        index_loosers = find(not(distances == min(distances)) & (distances - min(distances) <= 0.3));
        updateWinner = eta*(Xk_i - mean(:,index_winner));
        updateLoosers = updateWinner*(distances(index_winner)./distances(index_loosers)).^2;
        mean(:,index_winner) = mean(:,index_winner) + updateWinner;
        mean_aux = mean(:,index_loosers);
        mean(:,index_loosers) = mean(:,index_loosers) + updateLoosers;
    end
end

dist_tr = zeros(1,length(X(:,1)));
dist_val = zeros(1,length(X(:,1)));

for k = 1:length(X(:,1))
    Xi = X(k,:);
    Xi_val = X_val(k,:);
    distances = zeros(1,n); % obtain distances from x to each RBF center
    distances_val = zeros(1,n);
    for j = 1:n
        distances(j) = sqrt(sum((Xi'-mean(:,j)).^2));
        distances_val(j) = sqrt(sum((Xi_val'-mean(:,j)).^2));
    end
    dist_tr(k) = min(distances);
    dist_val(k) = min(distances_val);
end

avg_dist_tr(n-1) = sum(dist_tr)/length(dist_tr);
avg_dist_val(n-1) = sum(dist_val)/length(dist_val);

end

hold on
plot(nMin:nMax,avg_dist_tr,'r')
plot(nMin:nMax,avg_dist_val,'b')
hold off

