%% Time Series Prediction

clear
close all
% data

x = [1.5];

for i = 1:2000
    if i-1<25
        x(end+1) = 0.9*x(i);
    else
        x(end+1) = 0.9*x(i) + 0.2*x(i-25)/(1+x(i-25)^10);
    end
end

t = 301:1500;
sigma = 0.09; % std desviation random noise added
input = [x(t-20); x(t-15); x(t-10); x(t-5); x(t)] + sigma*randn(5,1200);
output = x(t+5);

plot(x,'b')
title('Time series')
xlabel('time')
ylabel('x(t)')
axis([301 1500 min(x)-0.1 max(x)+0.1 ])

% NN creation
nLayers = 3;
nHidden = [6 6]; % Number of hidden nodes per layer (if more than 2 hiddenLayers => row vector)

net = feedforwardnet(nHidden,'traingdx'); % training: backpropagation with momentum
net.divideFcn = 'divideind'; % Function for early stopping (separation training, val and test)
net.divideParam.trainInd = 1:800; % Adjust use of inputs
net.divideParam.valInd = 801:1000;
net.divideParam.testInd = 1001:1200;
net.divideParam.trainInd
net.numInputs = 5; % configuration number of inputs
for l = 1:5
    net.inputs{l}.size = 1;
end
net.inputConnect = [1 1 1 1 1;0 0 0 0 0;0 0 0 0 0];
net.numLayers = nLayers; % Configuration number of layers
net.layers{2}.transferFcn = 'tansig'
net.performParam.regularization = 0%Regularization
net.trainParam.max_fail = 1; % maximum validation fails to stop learning
net.trainParam.lr_inc = 1.02;
net.trainParam.show = 1;
net.trainParam.lr = 0.001;
net.trainParam.epochs = 100000;
net.trainParam.goal = 0.05;

tic;
net = train(net,input,output,'useParallel','yes');
t_training = toc
output_NN = net(input);
figure
hold on
plot(t(1001:1200),output(1001:1200),'k')
plot(t(1001:1200),output_NN(1001:1200),'r')
title('target vs predicted values of x(t+5)')
xlabel('time')
ylabel('x(t+5)')
legend('Target','Predicted','Location','SouthWest')
axis([1301 1500 min([output output_NN])-0.1 max([output output_NN])+0.1])
hold off

figure
plot(t,abs(output-output_NN),'LineWidth',1.5)
axis([301 1500 min([output output_NN])-0.1 max([output output_NN])+0.1])
xlabel('time')
ylabel('|out - target|')
title('absolute error in the prediction')

W = [];
V = [net.LW{2}(:)' net.LW{6}(:)'];

for l = 1:5
    W = [W net.IW{l}];
end

weights = [W(:)' V];
figure
histogram(weights,20)


