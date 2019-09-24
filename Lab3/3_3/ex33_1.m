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
    
    % Train network with d1, d2 and d3
    W = zeros(d);
    
    for i = 1:3
        W = W + (1/3)*data(i,:)'*data(i,:);
    end
    
    % W = W - eye(d);
    
    % Energy patterns p1, p2 and p3
    
    E = zeros(1,3); % energies p1,p2,p3
    
    for kk = 1:3
        E(kk) = -data(kk,:)*W*data(kk,:)';
    end
    
    E_dist = zeros(1,2);
    for kk = 10:11
        E_dist(kk-9) = -data(kk,:)*W*data(kk,:)';
    end
    E
    E_dist
    
%     x10 = data(10,:); % input pattern 10
%     x11 = data(11,:); % input pattern 11
%     
%     % introducing pattern 10 to the Hopfield NN
%     nUnchanged = 0;
%     while nUnchanged <100
%         index = randperm(d);
%         previous = x10;
%         for j = 1:d
%             x10(index(j)) = sign(W(index(j),:)*x10');
%         end
%         if sum(previous - x10) == 0
%             nUnchanged = nUnchanged + 1;
%         else
%             nUnchanged = 0;
%         end
%     end
%     
%     % Plot p10
%     figure(1)
%     imshow(mat2gray(reshape(x10,32,32)),'InitialMagnification', 1000);
%     xlabel('p10 after NN')
%     
%     % introducing pattern 11 to the Hopfield NN
%     nUnchanged = 0;
%     for n = 1:1000
%         index = randperm(d);
%         previous = x11;
%         for j = 1:d
%             x11(index(j)) = sign(W(index(j),:)*x11');
%         end
%         if sum(previous - x11) == 0
%             nUnchanged = nUnchanged + 1;
%         else
%             nUnchanged = 0;
%         end
%         if mod(n,100) == 0
%             figure(n)
%             imshow(mat2gray(reshape(x11,32,32)),'InitialMagnification', 1000);
%             xlabel(strcat('p11 after ',num2str(n),' iterations'))
%         end
%     end
%     
%     % Plot p10
%     figure(2)
%     imshow(mat2gray(reshape(x11,32,32)),'InitialMagnification', 1000);
%     xlabel('p11 after NN')
    
    