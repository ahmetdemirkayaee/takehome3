clear
close all;

n=7;
N_test=10000;
N_train=100;

a=n*rand(n,1);
mu = rand(n,1);
covariance = 0.2*rand(n,n) + eye(n);
covariance = covariance'*covariance; %To make it symmetric

%draw samples
x_training = mvnrnd(mu, covariance, N_train)';
x_test = mvnrnd(mu, covariance, N_test)';

v_training = mvnrnd(0, 1, N_train)';
v_test = mvnrnd(0, 1, N_test)';

loglikelihood = [];
loglikelihood_test = [];
optimal_beta_values = []

for alpha = 10.^(-3:0.05:3)*trace(covariance)/n
    %Alpha covariance and zero mean
    alpha,
    mu_alpha = zeros(n,1);
    covariance_alpha  = alpha*eye(n);

    z_training = mvnrnd(mu_alpha, covariance_alpha, N_train)'; %Alpha
    z_test = mvnrnd(mu_alpha, covariance_alpha, N_test)';      %Alpha

    %Find y such that y = aT(x+z) + v
    y_training = a.'*(x_training+z_training) + v_training;
    y_test = a.'*(x_test+z_test) + v_test;
        
    %Perform 10 fold cross validation to find optimal value of Beta hyperparameter 
    step_counts = 100;
	K = 10;
    mse_results = zeros(step_counts, K);
    betas = logspace(-4,3, step_counts);
    for beta = betas
        for k = 1:K
            y_validation_split = y_training(:,floor(((k-1)/K)*N_train)+1:floor(k/K*N_train));
            y_training_split = [y_training(:,1:floor(((k-1)/K)*N_train)),y_training(:,floor(k/K*N_train)+1:N_train)];

            x_validation_split = x_training(:,floor(((k-1)/K)*N_train)+1:floor(k/K*N_train));
            x_training_split = [x_training(:,1:floor(((k-1)/K)*N_train)),x_training(:,floor(k/K*N_train)+1:N_train)];

            x_training_split_with_ones = [ones(1, size(x_training_split,2)) ; x_training_split];
            
            w_hat = inv((x_training_split_with_ones*x_training_split_with_ones') + (eye(1)^2/beta^2)*eye(size(x_training_split_with_ones,1)) )*(y_training_split*x_training_split_with_ones')';%%%%sigmanin 

            x_validation_split_with_ones = [ones(1, size(x_validation_split,2)) ; x_validation_split];
            
            mse_results(find(logspace(-4,3, step_counts)==beta),k) = mean(mean((y_validation_split - w_hat'*x_validation_split_with_ones).^2));
        end
        mse_results_averaged_over_folds = mean(mse_results,2);
    end
    [min_mse_obtained, optimal_beta_value_ind]= min(mse_results_averaged_over_folds);
    optimal_beta_value = betas(optimal_beta_value_ind);
    optimal_beta_values = [optimal_beta_values optimal_beta_value];
    x_training_with_ones = [ones(1, size(x_training,2)) ; x_training];       
    w_hat = inv((x_training_with_ones*x_training_with_ones') + (eye(1)^2/optimal_beta_value^2)*eye(size(x_training_with_ones,1)) )*(y_training*x_training_with_ones')';%%%%sigmanin 
    loglikelihood = [loglikelihood mean(-(1/2)*(y_training - w_hat'*x_training_with_ones).^2)];

	x_test_with_ones = [ones(1, size(x_test,2)) ; x_test];       
    
    loglikelihood_test = [loglikelihood_test mean((y_test - w_hat'*x_test_with_ones).^2)];
    if alpha == 1*trace(covariance)/n
        figure
        stem((y_test - w_hat'*x_test_with_ones).^2);
        xlabel('Samples');
        ylabel('loglikelihood');        
        saveas(gcf,'alltest.png');
    end
end

figure(5);
loglog(10.^(-3:0.05:3)*trace(covariance)/n, loglikelihood, 'DisplayName','Training loglikelihood')
hold on
loglog(10.^(-3:0.05:3)*trace(covariance)/n, loglikelihood_test, 'DisplayName' , 'Test loglikelihood')
xlabel('\alpha')
grid on
legend('Location','NorthWest');
saveas(gcf,'q2Trainingvstestmse.png');

figure(6);
loglog(10.^(-3:0.05:3)*trace(covariance)/n, optimal_beta_values, 'r.')
xlabel('\alpha')
ylabel('\beta_{opt}')
grid on
title('Optimal \beta value for each \alpha');
saveas(gcf,'q2Optimalbetavalueforeachalpha.png');

function g = evalGaussian(x,mu,Sigma)
    %Function implemented by Prof. Deniz Erdogmus
    % Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
    [n,N] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    (x-repmat(mu,1,N))
    E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
    g = g';
end

