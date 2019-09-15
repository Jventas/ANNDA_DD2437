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

n = 18; % Number of RBF

mean = rand(2,n,'double'); % n uniform distribute random numbers in [0 1]
sigma = 0.2; % Sigma

deltaErr = 1000;
mean_initial = mean; % vector to save initial values of mean;

% Calculaiton of RBF centers
eta = 0.01;
for epoch = 1:1000 % 100 epochs enough
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
hold on
plot(X(:,1),X(:,2),'ok')
plot(mean(1,:),mean(2,:),'ob','MarkerFaceColor','b')
plot(mean_initial(1,:),mean_initial(2,:),'or','MarkerFaceColor','r')
plot(X_val(:,1),X_val(:,2),'*k')
title('RBF positions')
legend('Input Samples','RBF positions','Initial RBF positions','Validation samples','Location','NorthWest')
axis([0 1 0 1])
grid on
hold off

%% Delta Rule

eta = 0.0002;
w = 0.2*randn(n,2);
deltaW = zeros(n,2);

err_epochs = zeros(2,50000);
err_epochs_tr = zeros(2,50000);
for epoch = 1:50000
    index = randperm(length(X(:,1)));
    x_sf = X(index,:);
    f_sf = T(index,:);
    for i=1:length(x_sf)
        phi = zeros(1,n);
        for ii = 1:n
            phi(ii) = GaussianMulti(x_sf(i,1),x_sf(i,2),mean(1,ii),mean(2,ii),sigma,sigma);
        end
        err = f_sf(i,:) - phi*w;
        deltaW = eta*phi'*err;
        w = w + deltaW;
    end
    % Validation
    phi_val = zeros(length(X_val(:,1)),n);
    phi_tr = zeros(length(X_val(:,1)),n);
    for input_i = 1:length(X_val(:,1))
        for ii = 1:n
            phi_val(ii) = GaussianMulti(X_val(input_i,1),X_val(input_i,2),mean(1,ii),mean(2,ii),sigma,sigma);
        end
    end
    
    for input_i = 1:length(X(:,1))
        for ii = 1:n
            phi_tr(ii) = GaussianMulti(X(input_i,1),X(input_i,2),mean(1,ii),mean(2,ii),sigma,sigma);
        end
    end
    
    avg_err = sum(abs(T_val-phi_val*w),1)/100;
    avg_err_tr = sum(abs(T-phi_tr*w),1)/100;
    err_epochs(1,epoch) = avg_err(1);
    err_epochs(2,epoch) = avg_err(2);
    err_epochs_tr(1,epoch) = avg_err_tr(1);
    err_epochs_tr(2,epoch) = avg_err_tr(2);
end

figure(2)
plot(err_epochs','LineWidth',1.5)
grid on
xlabel('epochs')
ylabel('absolute error')
title('Validation Error')
legend('Validation error in output 1 (distance)','Validation error in output 2 (height)', ...
    'Location','Best')

figure(3)
plot(err_epochs_tr','LineWidth',1.5)
grid on
xlabel('epochs')
ylabel('absolute error')
title('Training Error')
legend('Training error in output 1 (distance)','Training error in output 2 (height)', ...
    'Location','Best')

figure(4)
plot(sqrt(sum(err_epochs_tr.^2,1)),'LineWidth',1.5)
grid on
title('Training Error Norm')
xlabel('epochs')
ylabel('||e||')