close all;
clear;
clc;

% Parameters
mu = [0.05, 0.02]; % Example values for µ1 and µ2
sigma = 0.04; % Example value for σ
S0 = 100; % Initial stock price
T = 1; % Time horizon (1 year)
dt = 0.01; % Time step
N = T / dt; % Number of time steps

% Time vector
t = linspace(0, T, N);

% Initialize stock price vector
S = zeros(1, N);
S(1) = S0;

% Generate S(t)
for i = 1:N - 1
    dW = sqrt(dt) * randn; % Increment of Wiener process
    S(i + 1) = S(i) + sum(mu .* S(i) .^ (1:length(mu)) * dt) + sigma * S(i) * dW;
end

% Load real data
data = readtable('all_stocks_5yr.csv');
real_prices = data.close; % 'close' prices are what we model

% Split data
split_ratio = 0.8;
idx = floor(split_ratio * length(real_prices));
train_data = real_prices(1:idx);
test_data = real_prices(idx + 1:end);

% Loop through different polynomial degrees
for k = 2:5
    fprintf('Testing with polynomial degree k = %d\n', k);

    % Prepare training and testing matrices
    X_train = zeros(length(train_data) - 1, k);
    X_test = zeros(length(test_data) - 1, k);

    for j = 1:k
        X_train(:, j) = train_data(1:end - 1) .^ j;
        X_test(:, j) = test_data(1:end - 1) .^ j;
    end

    Y_train = train_data(2:end);
    Y_test = test_data(2:end);

    % Normalize the features
    mu_X_train = mean(X_train);
    sigma_X_train = std(X_train);
    X_train_sgd = (X_train - mu_X_train) ./ sigma_X_train;
    X_test_sgd = (X_test - mu_X_train) ./ sigma_X_train;

    % Fit model on training data using LSQ
    trained_params = LSQ(X_train, Y_train);

    % Test Recursive LSQ on training data
    lambda_rls = 0.999; % Forgetting factor
    [rls_params, ~] = recursiveLSQ(X_train, Y_train, lambda_rls);

    % Define the model function

    model_fun = @(b, x) sum(x * b, length(b));

    % Fit model on training data using Gradient Descent
    alphas = [1e-11, 1e-10, 1e-9, 1e-8];
    % fprintf('Testing with alpha = %.4e\n', alphas);
    num_iters_values = round(linspace(1000, 4000, 4));
    % fprintf('Testing with num_iters = %d\n', num_iters_values);
    best = Inf;

    for alpha = alphas

        for num_iters = num_iters_values
            theta = rls_params; % Use RLS parameters as initial guess
            [trained_params_sgd, ~] = gradientDescent(X_train_sgd, Y_train, theta, alpha, num_iters);

            % Calculate mean squared error on test set
            mse_sgd = mean((model_fun(trained_params_sgd, X_test) - Y_test) .^ 2);

            if mse_sgd < best
                best = mse_sgd;
                best_alpha = alpha;
                best_num_iters = num_iters;
                fprintf('Best alpha: %.4e\n', best_alpha);
                fprintf('Best num_iters: %d\n', best_num_iters);
            end

        end

    end

    theta = rls_params; % Use RLS parameters as initial guess
    [trained_params_sgd, J_history] = gradientDescent(X_train_sgd, Y_train, theta, alpha, num_iters);

    % Predict on test data
    predicted_prices = model_fun(trained_params, X_test);
    predicted_prices_rls = model_fun(rls_params, X_test);
    predicted_prices_sgd = model_fun(trained_params_sgd, X_test);

    % Calculate mean squared error on test set
    mse = mean((predicted_prices - Y_test) .^ 2);
    mse_rls = mean((predicted_prices_rls - Y_test) .^ 2);
    mse_sgd = mean((predicted_prices_sgd - Y_test) .^ 2);

    fprintf('Mean Squared Error (LSQ): %.4f\n', mse);
    fprintf('Mean Squared Error (RLS): %.4f\n', mse_rls);
    fprintf('Mean Squared Error (SGD): %.4f\n', mse_sgd);

    figure;
    plot(Y_test(:, 1), 'b');
    hold on;
    plot(predicted_prices(:, 1), 'r');
    plot(predicted_prices_sgd(:, 1), 'g');
    plot(predicted_prices_rls(:, 1), 'm');
    legend('Actual Prices', 'LSQ Predictions', 'SGD Predictions', 'RLS Predictions');
    title(['Stock Price Predictions for k = ', num2str(k)]);
    xlabel('Time');
    ylabel('Price');
    axis tight;
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
