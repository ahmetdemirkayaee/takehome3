clear
close all

%create data
class_priors = [0.25 0.25 0.25 0.25 ];%Class priors
class_counts = 4 ;% number of Classes

scaler = 0.5;
dimensions = 3;
C(:,:,1) = eye(3)*0.15;
C(:,:,2) = eye(3)*0.40;
C(:,:,3) = eye(3)*0.05;
C(:,:,4) = eye(3)*0.35;

mu(:,:,1) = [1,0,0];
mu(:,:,2) = [0,1,0];
mu(:,:,3) = [0,0,1];
mu(:,:,4) = [1,1,0];

%==========Generate Data============%
for N = [100, 200, 500, 1000, 2000, 5000, 100000]
    D = generate_dataset(N, class_priors, dimensions , mu, C);
end

%=========Visualize Data============%
for N = [100, 200, 500, 1000, 2000, 5000, 100000]
    if ~isfile(strcat('d_',string(N), '.png'))
    
        figure(N);
        D = generate_dataset(N, class_priors, dimensions , mu, C);
        for class = 1:4
            data_for_class = D(D(:,dimensions+1)==class,1:dimensions);
            scatter3(data_for_class(:,1),data_for_class(:,2),data_for_class(:,3),'.');
            hold on   
        end
        xlabel('X_{1}')
        ylabel('X_{2}')
        zlabel('X_{3}')
        grid on
        title(strcat('Visualization of the data D^{', string(N), '}'));
        legend('Class 0', 'Class 1','Class 2', 'Class 3');
        saveas(gcf,strcat('d_',string(N), '.png'));
    end
end

%=============Theoretically optimal classifier===============%
D = generate_dataset(100000, class_priors, dimensions , mu, C);
likelihood = zeros(4, size(D,1));
for class_labels = 1:4
    likelihood(class_labels,:) = evalGaussian(D(:,1:3)', mu(1,:,class_labels)', C(:,:,class_labels)); %%%%%%%%%%%%% emin degilim
end
class_priors*likelihood;
[~,predicted_labels] = max(likelihood);
optimal_performance = sum(D(:,dimensions+1)==predicted_labels.')/length(predicted_labels)
%0.817620000000000

%%
% 
% %===========Model order selection====%
K = 10; %total_number_of_folds
max_number_of_neurons = 20;

validation_performance = zeros(6,max_number_of_neurons,K);
temp = 0
if ~isfile('validation_performance.mat')
    for N =  [100, 200, 500, 1000, 2000, 5000] %All training sets
        N,
        temp = temp + 1; %using to keep the order of N 
        D = generate_dataset(N, class_priors, dimensions , mu, C);
        for neurons = 1:max_number_of_neurons
            neurons,
            for k = 1:K %Fold
                k,
                D_validation = D(floor(((k-1)/K)*N)+1:floor(k/K*N),:);
                D_training = [D(1:floor(((k-1)/K)*N),:);D(floor(k/K*N)+1:N,:)];
                neural_net = patternnet(neurons);
                labels = zeros(N*0.9, 4);
                for n = 1:N*0.9
                    labels(n,D_training(n,dimensions+1)) = 1;
                end
                neural_net = train(neural_net, D_training(:,1:dimensions).', labels.');

                validation_labels = zeros(N*0.1, 4);
                for n = 1:N*0.1
                    validation_labels(n,D_validation(n,dimensions+1)) = 1;
                end
                validation_predictions = neural_net(D_validation(:,1:dimensions).');
                [~,validation_predicted_labels] = max(validation_predictions);
                validation_performance(temp, neurons, k) = sum(D_validation(:,dimensions+1)==validation_predicted_labels.')/length(validation_predicted_labels);
                %         	neural_net.numLayers = 3;
    %             neural_net.layers{2}.size = 8;
    %             neural_net.layers{2}.transferFcn = 'tansig';
            end
%             break
        end
%         break
    end
    save('validation_performance.mat','validation_performance')
else
    validation_performance = load('validation_performance.mat','validation_performance');
    validation_performance = validation_performance.validation_performance;
end
val_performance_folds_averaged = sum(validation_performance,3)/length(1:2:max_number_of_neurons);
optimal_perceptron_for_N = zeros(1,6);
for n = 1:6
    [~,optimal_neuron] = max(val_performance_folds_averaged(n,:));
    optimal_perceptron_for_N(1,n) = optimal_neuron;
end


%print test results for best neurons
figure(7);
experiment_count = 3; %try more than experiment and get the best in order to not stuch in local minima
test_performance = zeros(experiment_count,max_number_of_neurons);
N =  [100, 200, 500, 1000, 2000, 5000];
D_test = generate_dataset(100000, class_priors, dimensions , mu, C);        
test_labels = zeros(100000, 4);
for n = 1:100000
    test_labels(n,D_test(n,dimensions+1)) = 1;
end
for exp = 1:experiment_count
    temp = 1
    for n = N
        D_training = generate_dataset(n, class_priors, dimensions , mu, C);
        labels = zeros(n, 4);
        for index = 1:n
            labels(index,D_training(index, dimensions+1)) = 1;
        end
                
        %Train on whole data before test
        neural_net = patternnet(optimal_perceptron_for_N(temp));
        neural_net = train(neural_net, D_training(:,1:dimensions).' , labels.');
        test_predictions = neural_net(D_test(:,1:dimensions).');
        [~,test_predicted_labels] = max(test_predictions);
        test_performance(temp, exp) = sum(D_test(:,dimensions+1)==test_predicted_labels.')/length(test_predicted_labels);
        temp = temp + 1;
    end
end    
test_performance = max(test_performance')
semilogx(N,1-test_performance, 'bo')
hold on
semilogx(N,1-optimal_performance, 'ro')
xlabel('N')
ylabel('Probability of Error')
grid on
title('Test performance at Optimal perceptron for each dataset size');
legend('Experimental Results', 'Theoretical Minimum', 'Location','NorthEast');
saveas(gcf,'test_optimal_perceptron_dataset.png');




%optimal neurons for each dataset size:
figure(5);
N =  [100, 200, 500, 1000, 2000, 5000]
semilogx(N,optimal_perceptron_for_N, 'ro')
xlabel('N')
ylabel('Perceptrons')
grid on
title('Optimal perceptron for each dataset size');
% legend('Location','NorthEast');
saveas(gcf,'optimal_perceptron_dataset.png');





% if ~isfile(strcat('d_',string(N), '.png'))

figure(2);
N =  [100, 200, 500, 1000, 2000, 5000]
markers = {'bo-', 'r+-','mo-','*-','x-','v-','d-','^-','s-','>-'}

for neurons = 1:2:max_number_of_neurons
    semilogx(N,1-val_performance_folds_averaged(:,neurons), markers{(neurons-1)/2+1}, 'DisplayName',strcat(string(neurons), ' Neurons'))
    hold on
end
xlabel('N')
ylabel('Probability of Error')
grid on
% title(strcat('Visualization of the data '));
legend('Location','NorthEastOutside');
saveas(gcf,strcat('neurons.png');

% %=========Visualize Data============%
for N = [100, 200, 500, 1000, 2000, 5000, 100000]
    if ~isfile(strcat('d_',string(N), '.png'))
    
        figure(N);
        D = generate_dataset(N, class_priors, dimensions , mu, C);
        for class = 1:4
            data_for_class = D(D(:,dimensions+1)==class,1:dimensions);
            scatter3(data_for_class(:,1),data_for_class(:,2),data_for_class(:,3),'.')
            hold on   
        end
        xlabel('X_{1}')
        ylabel('X_{2}')
        grid on
        title(strcat('Visualization of the data D^{', string(N), '}'));
        saveas(gcf,strcat('d_',string(N), '.png'));
    end
end




function D = generate_dataset(N, class_priors, dimensions , mu, C)
    if ~isfile(strcat('question1_data_', string(N), '.mat'))
        D = zeros(N, dimensions+1);
        size(repmat(rand(1,N),length(class_priors),1))
        size(repmat(cumsum(class_priors.'),1,N))
        labels = repmat(rand(1,N),length(class_priors),1)> repmat(cumsum(class_priors.'),1,N);
        labels = sum(labels,1) + 1;
        class_counts = [sum(labels==1) sum(labels==2) sum(labels==3) sum(labels==4)];
        D(:,dimensions+1) = labels;
        for i = 1:4
            D(D(:,dimensions+1) == i, 1:dimensions) = mvnrnd(mu(:,:,i), C(:,:,i), class_counts(i));
        end
        save(strcat('question1_data_', string(N), '.mat'), 'D')
    else
        disp('Loading dataset')
        D = load(strcat('question1_data_', string(N), '.mat'));
        D = D.D;
    end
end




function g = evalGaussian(x,mu,Sigma)
    %Function implemented by Prof. Deniz Erdogmus
    % Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
    [n,N] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
    g = g';
end

