% SOM animals
clear all

% load in animals.mat file
animals = importdata('data/animals.dat');
animals = reshape(animals, 84, 32)';
% load in animal names and strip of apostrophes
% Animal input (32x84)
names = importdata('data/animalnames.txt');
names = strip(names);
names = strip(names, "'");

% initialise weight matrix
W = rand(84,100);

% SOM algorithm
n_epochs = 20;
step_size = 0.2;

neighbourhood(1,1:n_epochs) = 50;
for i=2:length(neighbourhood)
    neighbourhood(i) = neighbourhood(i-1)*0.75;
end
neighbourhood = round(neighbourhood);
    

rand_idx = randperm(length(names));
animals = animals(rand_idx, :);
names = names(rand_idx,:);


for epoch=1:n_epochs
    % start with large neighbourhood then decrease
    % neighbour reach indices
    n_reach = neighbourhood(epoch);
    
    % train for each animal
    for i=1:length(names)
        % calculate winner
        distance = sum((W-animals(i,:)').^2,1);
        [~,winner] = min(distance);
        
        % find neighbours
        % create matrix of ones for neighbours
        % is the neighbourhood circular though???? - no
        max_n = winner + n_reach;
        min_n = winner - n_reach;
        neighbours = zeros(1,100);
        neighbours(max(1,min_n):min(100, max_n)) = 1;
        
        % update weights
        for j=1:length(neighbours)
            % if neighbour, update weights
            if(neighbours(j) == 1)
                W(:,j) = W(:,j) + step_size.*(animals(i,:)' - W(:,j));   
            end
        end
        
    end
end

% Find indices of best match for animals
for i=1:length(names)
    distance = sum((W-animals(i,:)').^2,1);
    [~,winner] = min(distance);
    final_pos(i) = winner;
end
[~, idx] = unique(final_pos);
final_order = names(idx);
% sortedmat = final_pose(idx,:);

% final order example
example = {'giraffe';'camel';'pig';'horse';'antelop';'kangaroo';'bat';'elephant';'rabbit';'rat';'skunk';'hyena';'dog';'lion';'cat';'ape';'bear';'walrus';'crocodile';'seaturtle';'frog';'penguin';'ostrich';'duck';'pelican';'spider';'beetle';'dragonfly';'grasshopper';'butterfly';'housefly';'moskito'};
    