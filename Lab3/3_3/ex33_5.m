%% Exercise 3.3
    close all
    clear all
    clc

    % NN parameters
     d = 1024; % Number of neurons
    
    % Import data
    data = importdata('pict.dat');
    data = reshape(data,d,11); % there are 11 1024-patterns
    data = data';
    
    % weight matrix with random N(0,1) values
    W = randn(d);
    W = 0.5*(W+W');
    
    x10 = data(10,:); % input pattern 10
    x11 = data(11,:); % input pattern 11
    
    Niterations = 7;
    
    E10 = zeros(1,Niterations*d);
    E11 = zeros(1,Niterations*d);
    
    % introducing pattern 10 to the Hopfield NN
    for n = 1:Niterations
        n
        index = randperm(d);
        previous = x10;
        for k = 1:d
            x10(index(k)) = Sign(W(index(k),:)*x10');
            E10((n-1)*d+k) = -x10*W*x10'; % Energy calculation
        end
    end
    
    % introducing pattern 11 to the Hopfield NN
    for n = 1:Niterations
        n
        index = randperm(d);
        previous = x11;
        for k = 1:d
            x11(index(k)) = Sign(W(index(k),:)*x11');
            E11((n-1)*d+k) = -x11*W*x11'; % Energy calculation
        end
    end
    
    % Plot Energies
    
    hold on
    plot(1:Niterations*d,E10,'r','LineWidth',1.5)
    plot(1:Niterations*d,E11,'b','LineWidth',1.5)
    title('Energy vs iterations')
    xlabel('iteration')
    ylabel('E(X)')
    legend('Energy p10','Energy p11','Location','NorthEast')
    grid on
    hold off
    
    