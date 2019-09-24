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
    
    x10 = data(10,:)'; % input pattern 10
    x11 = data(11,:)'; % input pattern 11
     
    out10 = Sign(W*data(10,:)')';
    for n = 1:10
        x10 = Sign(W*x10);
    end
    
    % Plot p10
    figure(1)
    imshow(mat2gray(reshape(x10,32,32)),'InitialMagnification', 1000);
    xlabel('p10 after 10 iterations')
    
    out11 = Sign(W*data(11,:)')';
    for n = 1:10
        x11 = Sign(W*x11);
    end
    
    % Plot p11
    figure(2)
    imshow(mat2gray(reshape(x11,32,32)),'InitialMagnification', 1000);
    xlabel('p11 after 10 iterations')
    
    