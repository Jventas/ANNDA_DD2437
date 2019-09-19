% SOM cities
clear all

% import data
data = importdata('data/cities.dat');
data = data(3:end,:);
data = char(data);
coordinates = str2num(data);

% initialise weights
W = rand(2,10);

% SOM algorithm
n_epochs = 30;
step_size = 0.4;
 
neighbourhood(1,1:n_epochs) = 2;
for i=2:length(neighbourhood)
    neighbourhood(i) = neighbourhood(i-1)*0.85;
end
neighbourhood = round(neighbourhood);
%neighbourhood(1,1:n_epochs) = [2,2,2,2,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0];
% fig = figure();
% ax = axes('Parent',fig);
% ax.XLim = [0 1];
% ax.YLim = [0 1];
% lnh = plot(ax,coordinates(:,1),coordinates(:,2), '*');
labels = {'1','2','3','4','5','6','7','8','9','10'};

for epoch=1:n_epochs
    % start with large neighbourhood then decrease
    % neighbour reach indices
    n_reach = neighbourhood(epoch);
    % shuffle indices for animals
%     rand_idx = randperm(10);
%     coordinates = coordinates(rand_idx, :);
    % train for each cities
    for i=1:10
        % calculate winner
        distance = sum((W-coordinates(i,:)').^2,1);
        [~,winner] = min(distance);
        
        % find neighbours
        % create matrix of ones for neighbours
        % is the neighbourhood circular though????
        max_n = winner + n_reach;
        min_n = winner - n_reach;
        neighbours = zeros(1,10);
        if(max_n > 10)
            neighbours(winner:10) = 1;
            neighbours(1:max_n-10) = 1;
        else
            neighbours(winner:max_n) = 1;
        end
        if(min_n < 1)
            neighbours(1:max_n) = 1;
            neighbours(10+min_n:10) = 1;
        else
            neighbours(min_n:winner) = 1;
        end
           
        
        % update weights
        for j=1:10
            % if neighbour, update weights
            if(neighbours(j) == 1)
                W(:,j) = W(:,j) + step_size.*(coordinates(i,:)' - W(:,j));   
            end
        end
    end
    
    % for each epoch plot the nodes to see any dead nodes
%     clf
%     hold on    
%     plot(coordinates(:,1),coordinates(:,2),'or')
%     plot(W(1,:), W(2,:),'*b')
%     text(W(1,:), W(2,:),labels)
%     axis ([0 1 0 1])
%     hold off
%     drawnow
%     pause(0.5)
end

mask = ones(1,10);
% find winners
for i=1:10
    distance = sum((W-coordinates(i,:)').^2,1) .* mask;
    [~,winner] = min(distance);
    final_pos(i) = winner;
    mask(winner) = nan;
    
end
[~, idx] = unique(final_pos);
final_order = coordinates(idx,:);

% plot tour
figure
plot(final_order(:,1), final_order(:,2),'-x')
axis([0 1 0 1])
title({['Travelling Salesman']; ['Epochs: ' num2str(n_epochs)]; ['step size: ' num2str(step_size)]})
ylabel('x1')
xlabel('x2')