% SOM MPs votes
clear all

% import data
votes = importdata('data/votes.dat');
votes = reshape(votes, 31, 349);
votes = votes';
% Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
party = importdata('data/mpparty.dat');
% Male=0, Female=1
gender = importdata('data/mpsex.dat');
gender = str2double(gender);
district = importdata('data/mpdistrict.dat');

% intialise weights (10x31x10)
W = rand(10,31,10);

% SOM algorithm
n_epochs = 20;
step_size = 0.2;

% sort out neighbourhood
% not totally sure on this for 3D weight matrix
% neighbourhood(1,1:n_epochs) = 5;
% for i=2:length(neighbourhood)
%     neighbourhood(i) = neighbourhood(i-1)*0.9;
% end
% neighbourhood = round(neighbourhood);
neighbourhood = [5,5,4,4,3,3,2,2,2,2,1,1,1,1,1,1,0,0,0,0];


for epoch=1:n_epochs
    % start with large neighbourhood then decrease
    % neighbour reach indices
    n_reach = neighbourhood(epoch);
    % train for each MP
    for mp=1:349
        % calculate winner (distance = 1x31x100)
        distance = sum((W-votes(mp,:)).^2,2);
        % reshape
        distance = reshape(distance, 10, 10);
        min_el = min(distance(:));
        [i_win, j_win] = find(distance==min_el);
        
        % find neighbours
        % Manhattan distance for neighbours in 2D
        i_range = [max(1,i_win - n_reach) min(10,i_win + n_reach)];
        j_range = [max(1,j_win - n_reach) min(10,j_win + n_reach)];
        
        % update weights
        for i=i_range
            for j=j_range
                W(i,:,j) = W(i,:,j) + step_size.*(votes(mp,:) - W(i,:,j));
            end
        end

    end
end

for mp=1:349
    % find winner for each
    distance = sum((W-votes(mp,:)).^2,2);
    % reshape
    distance = reshape(distance, 10, 10);
    min_el = min(distance(:));
    [winner(mp,1), winner(mp,2)] = find(distance==min_el);
end

% plot the votes
% add a little noise so they don't overlap coordinates
mean = 0;
std = 0.25;
for i=1:349
    win(i,1) = winner(i,1) + normrnd(mean,std);
    win(i,2) = winner(i,2) + normrnd(mean,std);
end

% plot the gender
figure(1)
subplot(1,2,1);
hold on
for i=1:349
    if gender(i) == 0
        plot(win(i,1),win(i,2),'b*')
    else
        plot(win(i,1),win(i,2), 'g*')
    end
end
h = zeros(2, 1);
h(1) = plot(NaN,NaN,'b*');
h(2) = plot(NaN,NaN,'g*');
legend(h, 'male','female');
title('MP Gender');


% plot district
subplot(1,2,2);
hold on
for i = 1:349
    x = win(i,1);
    y = win(i,2);
    if district(i)==1
        plot(x,y,'b*')
    elseif district(i)==2
        plot(x,y,'c*')
    elseif district(i)==3
        plot(x,y,'r*')
    elseif district(i)==4
        plot(x,y,'k+')
    elseif district(i)==5
        plot(x,y,'g*')
    elseif district(i)==6
        plot(x,y,'m*')
    elseif district(i)==7
        plot(x,y,'y*')
    elseif district(i)==8
        plot(x,y,'k*')
    elseif district(i)==9
        plot(x,y,'bo')
    elseif district(i)==10
        plot(x,y,'co')
    elseif district(i)==11
        plot(x,y,'r+')
    elseif district(i)==12
        plot(x,y,'ro')
    elseif district(i)==13
        plot(x,y,'go')
    elseif district(i)==14
        plot(x,y,'mo')
    elseif district(i)==15
        plot(x,y,'yo')
    elseif district(i)==16
        plot(x,y,'ko')
    elseif district(i)==17
        plot(x,y,'b>')
    elseif district(i)==18
        plot(x,y,'c>')
    elseif district(i)==19
        plot(x,y,'r>')
    elseif district(i)==20
        plot(x,y,'b+')
    elseif district(i)==21
        plot(x,y,'g>')
    elseif district(i)==22
        plot(x,y,'m>')
    elseif district(i)==23
        plot(x,y,'y>')
    elseif district(i)==24
        plot(x,y,'k>')
    elseif district(i)==25
        plot(x,y,'cd')
    elseif district(i)==26
        plot(x,y,'rd')
    elseif district(i)==27
        plot(x,y,'rd')
    elseif district(i)==28
        plot(x,y,'gd')
    else
        plot(x,y,'m*')
    end
end
title('MP Districts')


% plot party
figure(2)
hold on
for i = 1:349
    x = win(i,1);
    y = win(i,2);
    m_party = party.data(i);
    if m_party==0
        plot(x,y,'ro')
    elseif m_party==1
        plot(x,y,'b*')
    elseif m_party==2
        plot(x,y,'r*')
    elseif m_party==3
        plot(x,y,'co')
    elseif m_party==4
        plot(x,y,'k*')
    elseif m_party==5
        plot(x,y,'g*')
    elseif m_party==6
        plot(x,y,'m*')
    else
        plot(x,y,'y*')
    end
end
h = zeros(8, 1);
h(1) = plot(NaN,NaN,'ro');
h(2) = plot(NaN,NaN,'b*');
h(3) = plot(NaN,NaN,'r*');
h(4) = plot(NaN,NaN,'co');
h(5) = plot(NaN,NaN,'k*');
h(6) = plot(NaN,NaN,'g*');
h(7) = plot(NaN,NaN,'m*');
h(8) = plot(NaN,NaN,'y*');
legend(h, 'no party','m','fp','s','v','mp','kd','c');
title('MP Party')