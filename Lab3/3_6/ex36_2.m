%% Exercise 3.6

clear
clc
close all

% Simulation parameters

bias = 0:1:40; % different bias

activity = 0.01:0.02:0.09; % different % of activity
storedPatterns = zeros(length(activity),length(bias)); % here the number of stored patterns

% NN parameters

nPatterns = 300; % number of patterns introduced in W
d = 100; % number
for nn = 1:length(activity)
    for n = 1:length(bias)

        % Generation of 300 random patterns

        patterns = zeros(nPatterns,d); 
        for i = 1:nPatterns
            index = randperm(d);
            index = index(1:round(d*activity(nn))); % indexes of activated neurons in pattern i
            patterns(i,index) = 1; % activation of patterns
        end

        % Obtaining average activity
        ro = (1/(d*nPatterns))*sum(sum(patterns));

        % Matrix W (training with all patterns)
        W = zeros(d);
        for i = 1:nPatterns
            W = W + (patterns(i,:)'-ro)*(patterns(i,:)-ro);
        end

        % Checking how many patterns are stored
        nStored = 0;
        outs = zeros(nPatterns,d); % output patterns
        for i = 1:nPatterns
            p = patterns(i,:); % pattern i
            p = (0.5 + 0.5*Sign(W*p' - bias(n)))';
            error = sum(p-patterns(i,:));
            if error == 0
                nStored = nStored + 1;
            end
        end
        storedPatterns(nn,n) = nStored;
    end
end

plot(bias,storedPatterns,'LineWidth',1.5)
xlabel('bias (\theta)')
ylabel('Number of stored patterns')
title('Bias vs number of stored patterns')
legend('\rho = 0.01','\rho = 0.03','\rho = 0.05','\rho = 0.07','\rho = 0.09',...
    'Location','NorthEast')
grid on

