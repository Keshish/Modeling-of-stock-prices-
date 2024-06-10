close all;
clear;
clc;

% First create a test for the implemented model
% Parameters
mu = [0.05, 0.02]; % Example values for µ1 and µ2
sigma = 0.04;      % Example value for σ
S0 = 100;          % Initial stock price
T = 1;             % Time horizon (1 year)
dt = 0.01;         % Time step
N = T/dt;          % Number of time steps
k = length(mu);    % Degree of the model

% Time vector
t = linspace(0, T, N);

% Initialize stock price vector
S = zeros(1, N);
S(1) = S0;

% Generate S(t)
for i = 1:N-1
    dW = sqrt(dt) * randn;  % Increment of Wiener process
    S(i+1) = S(i) + sum(mu .* S(i).^(1:k) * dt) + sigma * S(i) * dW;
end

% % Plot simulated stock price
% figure;
% plot(t, S);
% title('Simulated Stock Price');
% xlabel('Time (years)');
% ylabel('Stock Price');

% Fitting the model to real data
data = readtable('all_stocks_5yr.csv');
real_prices = data.close;  % 'close' prices are what we model

% For simplicity, let's assume k=2
% Define the model function
model_fun = @(b, x) b(1) * x + b(2) * x.^2;

% Plot the model function over a range of typical values
% x_values = linspace(min(real_prices), max(real_prices), 100);
% plot(x_values, model_fun([0.01, 0.01], x_values)); % Use initial_params if changed
% title('Model Function Behavior');
% xlabel('Stock Price');
% ylabel('Modeled Price Change');

X = [real_prices(1:end-1), real_prices(1:end-1).^2];
Y = real_prices(2:end);

mu_X = mean(X);
sigma_X = std(X);
X_sgd = (X - mu_X) ./ sigma_X;

% print the first 5 values of X and Y
fprintf('X: %.4f, %.4f\n', X(1:5, :));
fprintf('Y: %.4f\n', Y(1:5));


% Fit model using manual least squares
lsq_params = LSQ(X, Y);

% Fit model using gradient descent
% alpha = 1e-11;
% num_iters = 1000;
% theta = randn(size(X_sgd, 2), 1) * 0.01;
% [gd_params, ~] = gradientDescent(X_sgd, Y, theta, alpha, num_iters);

% Output results
fprintf('Least Squares Parameters: %.4f, %.4f\n', lsq_params);
% fprintf('Gradient Descent Parameters: %.4f, %.4f\n', gd_params);

% Split data
split_ratio = 0.8;
idx = floor(split_ratio * length(real_prices));
train_data = real_prices(1:idx);
test_data = real_prices(idx+1:end);

% Prepare training and testing matrices
X_train = [train_data(1:end-1), train_data(1:end-1).^2];
Y_train = train_data(2:end);
X_test = [test_data(1:end-1), test_data(1:end-1).^2];
Y_test = test_data(2:end);


% Normalize the features
mu_X_train = mean(X_train);
sigma_X_train = std(X_train);
X_train_sgd = (X_train - mu_X_train) ./ sigma_X_train;
X_test_sgd = (X_test - mu_X_train) ./ sigma_X_train;

% Fit model on training data using LSQ
trained_params = LSQ(X_train, Y_train);


% Test Recursive LSQ on training data
lambda = 0.999; % Forgetting factor
[rls_params, ~] = recursiveLSQ(X_train, Y_train, lambda);

% Fit model on training data using Gradient Descent
alpha = 1e-10;
num_iters = 1000;
theta = rls_params; % Use RLS parameters as initial guess
[trained_params_sgd, J_history] = gradientDescent(X_train_sgd, Y_train, theta, alpha, num_iters);

% Predict on test data
predicted_prices = model_fun(trained_params, X_test);
predicted_prices_rls = model_fun(rls_params, X_test);
predicted_prices_sgd = model_fun(trained_params_sgd, X_test);

% Calculate mean squared error on test set
mse = mean((predicted_prices - Y_test).^2);
mse_rls = mean((predicted_prices_rls - Y_test).^2);
mse_sgd = mean((predicted_prices_sgd - Y_test).^2);

fprintf('Mean Squared Error (LSQ): %.4f\n', mse);
fprintf('Mean Squared Error (RLS): %.4f\n', mse_rls);
fprintf('Mean Squared Error (SGD): %.4f\n', mse_sgd);

figure;
plot(Y_test(:,1), 'b');
hold on;
plot(predicted_prices(:,1), 'r');
plot(predicted_prices_sgd(:,1), 'g');
plot(predicted_prices_rls(:,1), 'm');
legend('Actual Prices', 'LSQ Predictions', 'SGD Predictions', 'RLS Predictions');
title('Stock Price Predictions');
xlabel('Time');
ylabel('Price');
axis tight;


% create a for loop to test k=1, k=2, k=3, k=4, k=5
k_values = 1:5;
mse_values = zeros(1, length(k_values));
mse_values_rls = zeros(1, length(k_values));
mse_values_sgd = zeros(1, length(k_values));

% for i = 1:length(k_values)
%     % Define the model function
%     if k_values(i) == 1
%         model_fun = @(b, x) b(1) * x;
%         X_train = [train_data(1:end-1)];
%         X_test = [test_data(1:end-1)];
%     elseif k_values(i) == 2
%         model_fun = @(b, x) b(1) * x + b(2) * x.^2;
%         X_train = [train_data(1:end-1), train_data(1:end-1).^2];
%         X_test = [test_data(1:end-1), test_data(1:end-1).^2];
%     elseif k_values(i) == 3
%         model_fun = @(b, x) b(1) * x + b(2) * x.^2 + b(3) * x.^3;
%         X_train = [train_data(1:end-1), train_data(1:end-1).^2, train_data(1:end-1).^3];
%         X_test = [test_data(1:end-1), test_data(1:end-1).^2, test_data(1:end-1).^3];
%     elseif k_values(i) == 4
%         model_fun = @(b, x) b(1) * x + b(2) * x.^2 + b(3) * x.^3 + b(4) * x.^4;
%         X_train = [train_data(1:end-1), train_data(1:end-1).^2, train_data(1:end-1).^3, train_data(1:end-1).^4];
%         X_test = [test_data(1:end-1), test_data(1:end-1).^2, test_data(1:end-1).^3, test_data(1:end-1).^4];
%     elseif k_values(i) == 5
%         model_fun = @(b, x) b(1) * x + b(2) * x.^2 + b(3) * x.^3 + b(4) * x.^4 + b(5) * x.^5;
%         X_train = [train_data(1:end-1), train_data(1:end-1).^2, train_data(1:end-1).^3, train_data(1:end-1).^4, train_data(1:end-1).^5];
%         X_test = [test_data(1:end-1), test_data(1:end-1).^2, test_data(1:end-1).^3, test_data(1:end-1).^4, test_data(1:end-1).^5];
%     end
% 
%     k = k_values(i);
%     % X_train = [train_data(1:end-1), train_data(1:end-1).^2, train_data(1:end-1).^3, train_data(1:end-1).^4, train_data(1:end-1).^5];
%     % X_test = [test_data(1:end-1), test_data(1:end-1).^2, test_data(1:end-1).^3, test_data(1:end-1).^4, test_data(1:end-1).^5];
% 
%     % Fit model on training data using LSQ
%     trained_params = LSQ(X_train, Y_train);
% 
%     % Test Recursive LSQ on training data
%     lambda = 0.999; % Forgetting factor
%     [rls_params, ~] = recursiveLSQ(X_train, Y_train, lambda);
% 
%     % Normalize the features
%     mu_X_train = mean(X_train);
%     sigma_X_train = std(X_train);
%     X_train_sgd = (X_train - mu_X_train) ./ sigma_X_train;
%     X_test_sgd = (X_test - mu_X_train) ./ sigma_X_train;
% 
%     % Fit model on training data using Gradient Descent
%     alpha = 1e-10;
%     num_iters = 1000;
%     theta = rls_params; % Use RLS parameters as initial guess
%     [trained_params_sgd, J_history] = gradientDescent(X_train_sgd, Y_train, theta, alpha, num_iters);
% 
%     % Predict on test data
%     predicted_prices = model_fun(trained_params, X_test);
%     predicted_prices_rls = model_fun(rls_params, X_test);
%     predicted_prices_sgd = model_fun(trained_params_sgd, X_test);
% 
%     % Calculate mean squared error on test set
%     mse_values(i) = mean((predicted_prices - Y_test).^2);
%     mse_values_rls(i) = mean((predicted_prices_rls - Y_test).^2);
%     mse_values_sgd(i) = mean((predicted_prices_sgd - Y_test).^2);
% end

figure;
plot(k_values, mse_values, 'b');
hold on;
plot(k_values, mse_values_rls, 'r');
plot(k_values, mse_values_sgd, 'g');
legend('LSQ', 'RLS', 'SGD');
title('Mean Squared Error vs. Model Complexity');
xlabel('Model Complexity (k)');
ylabel('Mean Squared Error');
axis tight;

% Least Squares (LSQ) estimator
function thetaLS = LSQ(X, Y)
thetaLS = (X.' * X) \ X.' * Y;
end

% Gradient Descent function
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% Initialize some useful values
m = length(y);
J_history = zeros(num_iters, 1);
% Ensure theta is a column vector
if size(theta, 2) > 1
    theta = theta';
end

for iter = 1:num_iters
    h = X * theta;
    error = h - y;

    %theta = theta - (alpha / m) * (X' * error);
    theta = theta - (alpha / m) * (X' * error);

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);
end
end

function J = computeCost(X, y, theta)
m = length(y);
J = 1 / (2 * m) * sum((X * theta - y).^2);
end

function [theta, P] = recursiveLSQ(X, Y, lambda)
    % Initialization
    [m, n] = size(X);
    theta = zeros(n, 1);
    P = eye(n) * 1e6; % Large initial value for P (similar to a large prior variance)
    
    % Recursive Least Squares
    for t = 1:m
        x_t = X(t, :)';
        y_t = Y(t);
        
        % Compute Kalman gain
        K_t = P * x_t / (lambda + x_t' * P * x_t);
        
        % Update estimate
        theta = theta + K_t * (y_t - x_t' * theta);
        
        % Update covariance matrix
        P = (P - K_t * x_t' * P) / lambda;
    end
end
