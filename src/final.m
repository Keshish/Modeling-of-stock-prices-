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

k_max = 4; % Maximum degree of polynomial terms to consider
best_k = 1;
best_model = [];
best_cost = Inf;

costs = zeros(k_max, 3);

% Iterate over different values of k
for k = 1:k_max
    % Construct features for train data
    X_train = constructFeatures(train_data, k);
    [X_train_scaled, mu, sig] = featureScaling(X_train);
    Y_train = train_data(2:end);

    % Estimate parameters using LSQ
    thetaLSQ = LSQ(X_train_scaled(1:end - 1, :), Y_train);
    % fprintf('thetaLSQ: %.4f\n', thetaLSQ);

    % Compute cost
    costLSQ = computeCost(X_train_scaled(1:end - 1, :), Y_train, thetaLSQ);

    % Estimate parameters using Recursive LSQ
    lambda_rls = 0.999; % Forgetting factor
    [thetaRLSQ, ~] = recursiveLSQ(X_train_scaled(1:end - 1, :), Y_train, lambda_rls);
    % fprintf('thetaRLSQ: %.4f\n', thetaRLSQ);

    % Compute cost
    costRLSQ = computeCost(X_train_scaled(1:end - 1, :), Y_train, thetaRLSQ);

    % Estimate parameters using Gradient Descent
    alpha = 1e-11;
    num_iters = 1000;
    theta = thetaLSQ;
    [thetaGD, ~] = gradientDescent(X_train_scaled(1:end - 1, :), Y_train, theta, alpha, num_iters);

    % Compute cost
    costGD = computeCost(X_train_scaled(1:end - 1, :), Y_train, thetaGD);

    % % Print results
    % fprintf('Degree k = %d\n', k);
    % fprintf('LSQ: %.4f\n', costLSQ);
    % fprintf('RLSQ: %.4f\n', costRLSQ);
    % fprintf('GD: %.4f\n', costGD);

    % Save the costs for future plot
    costs(k, 1) = costLSQ;
    costs(k, 2) = costRLSQ;
    costs(k, 3) = costGD;

    bestEstimator = thetaLSQ;
    cost = costLSQ;

    if costRLSQ < cost
        bestEstimator = thetaRLSQ;
        cost = costRLSQ;
    end

    % Predict on train data
    Y_pred_train = X_train(1:end - 1, :) * bestEstimator;

    if cost < best_cost
        best_cost = cost;
        best_k = k;
        best_model = bestEstimator;
    end

end

% Plot costs
figure;
plot(1:k_max, costs(:, 1), 'b');
hold on;
plot(1:k_max, costs(:, 2), 'r');
plot(1:k_max, costs(:, 3), 'g');
legend('LSQ', 'RLSQ', 'GD');
title('Error vs. Degree of Polynomial');
xlabel('Degree of Polynomial');
ylabel('Cost');
hold off;


% Construct features for test data
X_test = constructFeatures(test_data, best_k);
X_test_scaled = (X_test(:, 1:end - 1) - mu) ./ sig;
X_test_scaled = [X_test_scaled, X_test(:, end)];
Y_test = test_data(2:end);

% Plot the best model
figure;
plot(Y_test, 'b');
hold on;
plot(X_test_scaled * best_model, 'r');
legend('Actual', 'Predicted');
title('Stock Price Prediction with Best Model');
xlabel('Time');
ylabel('Stock Price');
hold off;


% Print MSE and R^2
MSE = computeCost(X_test_scaled(1:end - 1, :), Y_test, best_model);
R2 = 1 - MSE / var(Y_test);

fprintf('Best model has degree k = %d\n', best_k);
fprintf('MSE: %.4f\n', MSE);
fprintf('R^2: %.4f\n', R2);

% Estimate sigma
sigma = best_model(end);

% Simulation parameters
T = length(test_data) - 1;
dt = 1; % Assuming daily data
S0 = test_data(1);

% Simulate the model for a single time step
sim_test = simulate(test_data(1), best_model, sigma, 2, 1, best_k, mu, sig);
fprintf('Simulated price: %.4f\n', sim_test);

% Simulate the stock prices
simulated_prices = simulate(S0, best_model, sigma, T, dt, best_k, mu, sig);

% Also simulate for each day with the previous day's data
simulated_prices_daily = zeros(T, 1);
simulated_prices_daily(1) = test_data(1);

for t = 2:T
    tmp = simulate(test_data(t - 1), best_model, sigma, 2, 1, best_k, mu, sig);
    simulated_prices_daily(t) = tmp(2);
end

% Simulate for every week
simulated_prices_weekly = zeros(T, 1);
simulated_prices_weekly(1) = test_data(1);

for t = 2:T

    if mod(t, 7) == 0
        tmp = simulate(test_data(t - 1), best_model, sigma, 2, 1, best_k, mu, sig);
        simulated_prices_weekly(t) = tmp(2);
    else
        simulated_prices_weekly(t) = simulated_prices_weekly(t - 1);
    end

end

% Print MSE and R^2
MSE_simulated = mean((simulated_prices - test_data(2:end)) .^ 2);
R2_simulated = 1 - MSE_simulated / var(test_data(2:end));

MSE_daily = mean((simulated_prices_daily - test_data(2:end)) .^ 2);
R2_daily = 1 - MSE_daily / var(test_data(2:end));

MSE_weekly = mean((simulated_prices_weekly - test_data(2:end)) .^ 2);
R2_weekly = 1 - MSE_weekly / var(test_data(2:end));

fprintf('MSE (Simulated): %.4f\n', MSE_simulated);
fprintf('R^2 (Simulated): %.4f\n', R2_simulated);

fprintf('MSE (Daily): %.4f\n', MSE_daily);
fprintf('R^2 (Daily): %.4f\n', R2_daily);

fprintf('MSE (Weekly): %.4f\n', MSE_weekly);
fprintf('R^2 (Weekly): %.4f\n', R2_weekly);

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

figure;
plot(test_data(2:end), 'b');
hold on;
plot(simulated_prices_weekly, 'r');
legend('Actual', 'Simulated');
title('Stock Price Prediction with Weekly Simulation');
xlabel('Time');
ylabel('Stock Price');
hold off;

sensitivityAnalysis(best_model, sigma, test_data, best_k, mu, sig);

function S = simulate(S0, theta, sigma, T, dt, k, mu, sig)
    N = round(T / dt);
    S = zeros(N, 1);
    S(1) = S0;

    for t = 2:N
        dW = sqrt(dt) * randn; % Wiener process increment
        drift = 0;

        % Compute drift term
        for i = 1:k
            % Scale the feature before using it in the model
            scaled_feature = (S(t - 1) ^ i - mu(i)) / sig(i);
            drift = drift + theta(i) * scaled_feature;
        end

        % fprintf('drift: %.4f\n', drift);
        drift = drift * dt;

        % Update stock price
        scaled_feature = (S(t - 1) - mu(k)) / sig(k);
        S(t) = S(t - 1) + drift * dt + sigma * scaled_feature * dW;

        %fprintf('Stochastic term: %.4f\n', sigma * scaled_feature * dW);

        % Convert negative stock prices to positive because stock prices cannot be negative
        S(t) = abs(S(t));
    end

end

% Feature Construction
function X = constructFeatures(data, k)
    n = length(data);
    X = zeros(n, k);

    for i = 1:k
        X(:, i) = data .^ i;
    end

    %Add sigma term
    X = [X, ones(n, 1)];
end

% Least Squares (LSQ) estimator
function thetaLS = LSQ(X, Y, lambda)

    if nargin < 3
        lambda = 1e-2;
    end

    thetaLS = (X.' * X + lambda * eye(size(X, 2))) \ X.' * Y;
end

function [X_scaled, mu, sigma] = featureScaling(X)
    % Exclude the last column (sigma) from scaling
    X_features = X(:, 1:end - 1);

    % Compute mean and standard deviation for each feature
    mu = mean(X_features);
    sigma = std(X_features);

    % Scale features
    X_features_scaled = (X_features - mu) ./ sigma;

    % Append the last column (sigma) without scaling
    X_scaled = [X_features_scaled, X(:, end)];
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
        % fprintf('h: %.4f\n', h);
        % fprintf('size of h: %.4f\n', size(h));
        % fprintf('size of y: %.4f\n', size(y));
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


function sensitivityAnalysis(best_model, sigma, test_data, best_k, mu, sig)
    % Define the range of variation for each parameter
    theta_variation = linspace(0.9, 1.1, 5); % Vary theta by ±10%
    sigma_variation = linspace(0.9, 1.1, 5); % Vary sigma by ±10%

    % Initialize arrays to store the results
    MSE_theta = zeros(length(best_model), length(theta_variation));
    MSE_sigma = zeros(1, length(sigma_variation));

    % Sensitivity analysis for theta
    for i = 1:length(best_model)
        for j = 1:length(theta_variation)
            % Vary the i-th parameter of theta
            theta_varied = best_model;
            theta_varied(i) = best_model(i) * theta_variation(j);

            % Simulate the stock prices with the varied parameter
            simulated_prices_varied = simulate(test_data(1), theta_varied, sigma, length(test_data) - 1, 1, best_k, mu, sig);

            % Compute the MSE
            MSE_theta(i, j) = mean((simulated_prices_varied - test_data(2:end)).^2);
        end
    end

    % Sensitivity analysis for sigma
    for j = 1:length(sigma_variation)
        % Vary sigma
        sigma_varied = sigma * sigma_variation(j);

        % Simulate the stock prices with the varied sigma
        simulated_prices_varied = simulate(test_data(1), best_model, sigma_varied, length(test_data) - 1, 1, best_k, mu, sig);

        % Compute the MSE
        MSE_sigma(j) = mean((simulated_prices_varied - test_data(2:end)).^2);
    end

    % Plot the results
    figure;
    for i = 1:length(best_model)
        subplot(length(best_model), 1, i);
        plot(theta_variation, MSE_theta(i, :), 'b');
        title(['Sensitivity Analysis for \theta_', num2str(i)]);
        xlabel('Parameter Variation');
        ylabel('MSE');
    end

    figure;
    plot(sigma_variation, MSE_sigma, 'r');
    title('Sensitivity Analysis for \sigma');
    xlabel('Parameter Variation');
    ylabel('MSE');
end
