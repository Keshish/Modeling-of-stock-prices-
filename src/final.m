close all;
clear;
clc;

data = readtable('all_stocks_5yr.csv');
real_prices = data.close; % 'close' prices are what we model

% Split data
split_ratio = 0.8;
idx = floor(split_ratio * length(real_prices));
train_data = real_prices(1:idx);
test_data = real_prices(idx + 1:end);

k_max = 3; % Maximum degree of polynomial terms to consider
best_k = 1;
best_model = [];
best_cost = Inf;

% Iterate over different values of k
for k = 1:k_max
    % Construct features for train data
    X_train = constructFeatures(train_data, k);
    Y_train = train_data(2:end);
    
    % Estimate parameters using LSQ
    thetaLS = LSQ(X_train(1:end-1, :), Y_train);
    
    % Predict on train data
    Y_pred_train = X_train(1:end-1, :) * thetaLS;
    
    % Compute cost
    cost = computeCost(X_train(1:end-1, :), Y_train, thetaLS);
    
    if cost < best_cost
        best_cost = cost;
        best_k = k;
        best_model = thetaLS;
    end
end

% Construct features for test data
X_test = constructFeatures(test_data, best_k);
Y_test = test_data(2:end);

% Predict on test data
Y_pred_test = X_test(1:end-1, :) * best_model;

% Print MSE and R^2
MSE = computeCost(X_test(1:end-1, :), Y_test, best_model);
R2 = 1 - MSE / var(Y_test);

fprintf('Best model has degree k = %d\n', best_k);
fprintf('MSE: %.4f\n', MSE);
fprintf('R^2: %.4f\n', R2);

% Estimate sigma
log_returns = log(train_data(2:end) ./ train_data(1:end-1));
sigma = std(log_returns);

% Simulation parameters
T = length(test_data) - 1;
dt = 1; % Assuming daily data
S0 = test_data(1);

% Simulate the stock prices
simulated_prices = simulate(S0, best_model, sigma, T, dt, best_k);

sim_test = simulate(52.3, best_model, sigma, 1, 1, best_k);
fprintf('Simulated price: %.4f\n', sim_test);

% Also simulate for each day with the previous day's data
simulated_prices_daily = zeros(T, 1);
for t = 2:T
    simulated_prices_daily(t) = simulate(test_data(t-1), best_model, sigma, 1, 1, best_k);
end

% Print MSE and R^2
MSE_simulated = mean((simulated_prices - test_data(2:end)) .^ 2);
R2_simulated = 1 - MSE_simulated / var(test_data(2:end));

fprintf('MSE (Simulated): %.4f\n', MSE_simulated);
fprintf('R^2 (Simulated): %.4f\n', R2_simulated);

% Plot results
figure;
plot(test_data, 'b');
hold on;
plot(simulated_prices, 'r');
legend('Actual', 'Simulated');
title('Stock Price Prediction with Stochastic Model');
xlabel('Time');
ylabel('Stock Price');
hold off;

figure;
plot(test_data(2:end), 'b');
hold on;
plot(simulated_prices_daily, 'r');
legend('Actual', 'Simulated');
title('Stock Price Prediction with Daily Simulation');
xlabel('Time');
ylabel('Stock Price');
hold off;


% Simulate the model
function S = simulate(S0, theta, sigma, T, dt, k)
    N = round(T / dt);
    S = zeros(N, 1);
    S(1) = S0;
    for t = 2:N
        dW = sqrt(dt) * randn; % Wiener process increment
        drift = 0;
        for i = 1:k
            drift = drift + theta(i) * S(t-1)^i;
        end
        % Ensure the drift term is not excessively large
        drift = min(max(drift, -1e2), 1e2);
        % Update stock price
        
        S(t) = S(t-1) + drift * dt + sigma * S(t-1) * dW;
        % Ensure non-negative stock price
        S(t) = max(S(t), 0);
    end
end

% Feature Construction
function X = constructFeatures(data, k)
    n = length(data);
    X = zeros(n, k);
    for i = 1:k
        X(:, i) = data .^ i;
    end
end

% Least Squares (LSQ) estimator
function thetaLS = LSQ(X, Y, lambda)
    if nargin < 3
        lambda = 1e-2;
    end
    thetaLS = (X.' * X + lambda * eye(size(X, 2))) \ X.' * Y;
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

        theta = theta - (alpha / m) * (X' * error);

        % Save the cost J in every iteration
        J_history(iter) = computeCost(X, y, theta);

        % Add stopping criterion
        if iter > 1 && abs(J_history(iter) - J_history(iter - 1)) < 1e-6
            break;
        end

    end

end

function J = computeCost(X, y, theta)
    m = length(y);
    J = 1 / (2 * m) * sum((X * theta - y) .^ 2);
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