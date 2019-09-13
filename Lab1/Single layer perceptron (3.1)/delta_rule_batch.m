%% Delta rule batch

clear
close all

% Data generation

n = 100; % Number of data
nInputs = 2;
nOutputs = 1;
mA = [ 0.75, 1]; sigmaA = 0.4; % Mean and std. dev of A
mB = [-0.75, -1]; sigmaB = 0.4; % Mean and std. dev of B

classA(1,:) = randn(1,n) .* sigmaA + mA(1);
classA(2,:) = randn(1,n) .* sigmaA + mA(2);
classB(1,:) = randn(1,n) .* sigmaB + mB(1);
classB(2,:) = randn(1,n) .* sigmaB + mB(2);

p = randperm(n);

% Input data

X = [classA classB];
index = randperm(2*n);

X(1,:) = X(1,index); % X => input matrix
X(2,:) = X(2,index);
X(3,:) = ones(1,2*n);

T = sign(n-index); % T => target matrix
T(T==0) = 1;

rate = 0.01; % learning rate or step size
W = 0.2*randn(nOutputs,nInputs+1); % Random initialization of W = [w1 w2 theta]

% Neccesary variables for plotting the decision boundary
x1 = -4:0.01:4;
x2 = W(3)/W(2)-W(1)*x1/W(2);

Y1 = W*X; % output before thresholding
Y = sign(Y1); % output after thresholding
e = 0.5*(T-Y1); %error

deltaW = [1000 1000 1000]; % initial deltaW
mse = [];
deltaErr = 1000;

while deltaErr>=1e-6
    clf('reset')
    Y1 = W*X; % output before thresholding
    e = -0.5*(T-Y1); % error calculation
    mse(end+1) = sum(e.^2)/length(e);
    deltaW = -rate*e*transpose(X); % change of deltaW
    W = W + deltaW; % update W
    x2 = W(3)/W(2)-W(1)*x1/W(2); % update x2 for plotting
    hold on
    plot(classA(1,:),classA(2,:),'or')
    plot(classB(1,:),classB(2,:),'*b')
    plot(x1,x2,'k','LineWidth',2)
    xlabel('x_{1}')
    ylabel('x_{2}')
    title('Delta Rule')
    legend('Class A','Class B','Decision boundary','Location','NorthWest')
    axis ([min(X(1,:))-0.5 max(X(1,:))+0.5 min(X(2,:))-0.5 max(X(2,:))+0.5])
    hold off
    drawnow
    if (length(mse)>1)
        deltaErr = abs(mse(end-1) - mse(end));
    end
end

%plot evolution of MSE (Mean Square Error)
figure
plot(mse,'g','LineWidth',1.5)
xlabel('epoch')
ylabel('mse')
title('Evolution of mse along epochs (Delta Rule)')
grid on
