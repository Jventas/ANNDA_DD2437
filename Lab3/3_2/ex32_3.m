%% Exercise 3.2
    close all
    clear all
    clc

    % NN parameters
     d = 1024; % Number of neurons
    
    % Import data
    data = importdata('pict.dat');
    data = reshape(data,d,11); % there are 11 1024-patterns
    data = data';
    
    % Train network with d1, d2 and d3
    W = zeros(d);
    
    for i = 1:3
        W = W + (1/3)*data(i,:)'*data(i,:);
    end
    
    W = W - eye(d);
    
    x10 = data(10,:); % input pattern 10
    x11 = data(11,:); % input pattern 11
    
    % introducing pattern 10 to the Hopfield NN
    nIter = 0;
    for n = 1:7
        index = randperm(d);
        previous = x10;
        for j = 1:d
            nIter = nIter + 1;
            x10(index(j)) = sign(W(index(j),:)*x10');
            if mod(nIter,100) == 0
                figure(nIter)
                imshow(mat2gray(reshape(x10,32,32)),'InitialMagnification', 1000);
                xlabel(strcat('p10 after ',num2str(nIter),' iterations'))
            end
        end
    end
    
    % introducing pattern 11 to the Hopfield NN
    nUnchanged = 0;
    nIter2 = 0;
    for n = 1:7
        index = randperm(d);
        previous = x11;
        for j = 1:d
            nIter2 = nIter2 + 1;
            x11(index(j)) = sign(W(index(j),:)*x11');
            if mod(nIter2,100) == 0
                figure(nIter + nIter2)
                imshow(mat2gray(reshape(x11,32,32)),'InitialMagnification', 1000);
                xlabel(strcat('p11 after ',num2str(nIter2),' iterations'))
            end
        end
    end
   
    