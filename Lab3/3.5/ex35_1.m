%% Exercise 3.5

% Is the same of 3.4 but
    close all
    clear all
    clc

    % NN parameters
     d = 1024; % Number of neurons
    
    % Import data
    data = importdata('pict.dat');
    data = reshape(data,d,11); % there are 11 1024-patterns
    data = data';
    
    % Train network with d1, d2, d3 and d4
    W = zeros(d);
    
    for i = 1:4
        W = W + data(i,:)'*data(i,:);
    end
    
   % W = W - eye(d);
    
    
    
    Niterations = 10;
    Nexperiments = 100;
    
    noise_samples = round((0.1:0.1:1)*d);
    nCorrect = zeros(length(noise_samples),3); 
    nCorrect_inv = zeros(length(noise_samples),3); 
    for nn = 1:Nexperiments
        for kk = 1:length(noise_samples)
            x1 = data(1,:); % input pattern 1
            x2 = data(2,:); % input pattern 2
            x3 = data(3,:); % input pattern 3
            % introducing pattern 1 to the Hopfield NN
            index1 = randperm(d);
            index1 = index1(1:noise_samples(kk));
            x1(index1) = -x1(index1);
            for n = 1:Niterations 
                x1 = (Sign(W*x1'))';
            end

            % introducing pattern 2 to the Hopfield NN
            index2 = randperm(d);
            index2 = index2(1:noise_samples(kk));
            x2(index2) = -x2(index2);
            for n = 1:Niterations
                x2 = (Sign(W*x2'))';
            end

            % introducing pattern 3 to the Hopfield NN
            index3 = randperm(d);
            index3 = index3(1:noise_samples(kk));
            x3(index3) = -x3(index3);
            for n = 1:Niterations
                x3 = (Sign(W*x3'))';
            end
            
            X = [x1;x2;x3];
            data_inv = -data(1:3,:);
            for n = 1:3
                nCorrect(kk,n) = nCorrect(kk,n) + (isequal(X(n,:),data(n,:)) + 0);
                nCorrect_inv(kk,n) = nCorrect_inv(kk,n) + (isequal(X(n,:),data_inv(n,:)) + 0);
            end
        end
    end
    
    ratioCorrect = nCorrect/Nexperiments;
    ratioCorrect_inv = nCorrect_inv/Nexperiments;
    
    percent = (10:10:100)';
    hold on
    plot(percent,ratioCorrect(:,1),'r','LineWidth',1.5)
    plot(percent,ratioCorrect_inv(:,1),'r--','LineWidth',1.5)
    plot(percent,ratioCorrect(:,2),'g','LineWidth',1.5)
    plot(percent,ratioCorrect_inv(:,2),'g--','LineWidth',1.5)
    plot(percent,ratioCorrect(:,3),'b','LineWidth',1.5)
    plot(percent,ratioCorrect_inv(:,3),'b--','LineWidth',1.5)
    xlabel('percentage of distorted data')
    title('Distortion Resistance (100 iterations) adding p4 to W')
    ylabel('ratio of correctly-recovered output data (100 iterations)')
    legend('p1','p1 symmetric','p2','p2 symmetric','p3','p3 symmetric', ...
        'Location','Best')
    grid on
    hold off
    
    
   
    
  
    
    