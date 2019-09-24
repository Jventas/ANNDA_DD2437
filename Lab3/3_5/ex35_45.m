%% Exercise 3.5 - 4

clear
close all

Iterations = 100;
d = 100; % Number of neurons (input dimension)

% exercise parameters
nPatterns = 300;
stablePatterns_tot = zeros(1,nPatterns);
recoveredPatterns_tot = zeros(1,nPatterns);

for nnn = 1:Iterations
    nnn
    % NN parameters

    W = zeros(d);
    patterns = zeros(nPatterns,d); % matrix to store random patterns
    stablePatterns = zeros(1,nPatterns);
    recoveredPatterns = zeros(1,nPatterns);

    for nn = 1:nPatterns
        % New random pattern
        randpattern = 2*(randi(2,[1,d])-1) - 1;
        patterns(nn,:) = randpattern;
        % patterns = unique(patterns,'rows');

        % Updating W
        W = W + randpattern'*randpattern;

        % Check how many patterns remain stable
        stables = 0;
        for n = 1:size(patterns,1)
            p = patterns(n,:); % get pattern
            p_out = (Sign(W*p'))';
            if isequal(p,p_out)
                stables = stables + 1;
            end
        end 
        stablePatterns(nn) = stables/size(patterns,1);
        
        % Check how many patterns can be recovered
        nRecovered = 0;
        for n = 1:size(patterns,1)
            index = randperm(d);
            index = index(1:5); % 5 flipped samples
            p = patterns(n,:); % get pattern
            p_dist = p;
            p_dist(index) = -p_dist(index);
            for k = 1:7
                if k == 1
                    p_out = (Sign(W*p_dist'))';
                else
                    p_out = (Sign(W*p_out'))';
                end
            end
            if isequal(p,p_out)
                nRecovered = nRecovered + 1;
            end
        end
        recoveredPatterns(nn) = nRecovered/size(patterns,1);
    end
    stablePatterns_tot = stablePatterns_tot + stablePatterns;
    recoveredPatterns_tot = recoveredPatterns_tot + recoveredPatterns;
    
end

stablePatterns_tot = stablePatterns_tot/Iterations;
recoveredPatterns_tot = recoveredPatterns_tot/Iterations;

% Plot results
hold on
plot(1:nPatterns,stablePatterns_tot,'b','LineWidth',1.5)
plot(1:nPatterns,recoveredPatterns_tot,'r','LineWidth',1.5)
xlabel('Number of patterns in W')
ylabel('ratio (stable patterns)/(total patterns)')
title('Stable and recovered patterns vs number of learnt patterns (average in 100 iterations)')
legend('Stable patterns','Recovered patterns','Location','Best')
grid on
hold off


