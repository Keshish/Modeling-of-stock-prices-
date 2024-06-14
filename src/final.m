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

% Prepare training and testing matrices
X_train = [train_data(1:end - 1)];
Y_train = train_data(2:end);
X_test = [test_data(1:end - 1)];
Y_test = test_data(2:end);

% Normalize the features
mu_X_train = mean(X_train);
sigma_X_train = std(X_train);
X_train_sgd = (X_train - mu_X_train) ./ sigma_X_train;
X_test_sgd = (X_test - mu_X_train) ./ sigma_X_train;

% Main script to estimate parameters for different k and test the model
num_iters = 1000; % Number of iterations for gradient descent
alpha = 0.01; % Learning rate for gradient descent
lambda = 0.99; % Forgetting factor for recursive LSQ
dt = 1; % Time increment (assumed to be 1 for simplicity)

for k = 2:5
    % Construct feature matrix for current k
    X_train = zeros(length(train_data) - 1, k);
    X_test = zeros(length(test_data) - 1, k);

    for j = 1:k
        X_train(:, j) = train_data(1:end - 1) .^ j;
        X_test(:, j) = test_data(1:end - 1) .^ j;
    end

    % Normalize features for gradient descent
    mu_X_train_k = mean(X_train);
    sigma_X_train_k = std(X_train);
    X_train_k_sgd = (X_train - mu_X_train_k) ./ sigma_X_train_k;
    X_test_k_sgd = (X_test - mu_X_train_k) ./ sigma_X_train_k;

    % LSQ Estimation
    theta_LSQ = LSQ(X_train, Y_train);

    % estimate sigma
    sigma = 1e-2;

    % Gradient Descent Estimation
    initial_theta = zeros(size(X_train, 2), 1);
    [theta_GD, J_history] = gradientDescent(X_train_k_sgd, Y_train, initial_theta, alpha, num_iters);

    % Recursive LSQ Estimation
    [theta_RLSQ, P] = recursiveLSQ(X_train, Y_train, lambda);

    % Simulate differential equation using LSQ estimates
    S0 = X_test(1);
    S_sim_LSQ = simulate_diff_eq(theta_LSQ, sigma, S0, dt, length(X_test));

    % Simulate differential equation using Gradient Descent estimates
    S_sim_GD = simulate_diff_eq(theta_GD, sigma, S0, dt, length(X_test));

    % Simulate differential equation using Recursive LSQ estimates
    S_sim_RLSQ = simulate_diff_eq(theta_RLSQ, sigma, S0, dt, length(X_test));

    % Compare simulated values with actual test data
    fprintf('For k = %d:\n', k);
    fprintf('LSQ estimates:\n');
    disp(theta_LSQ);
    fprintf('Gradient Descent estimates:\n');
    disp(theta_GD);
    fprintf('Recursive LSQ estimates:\n');
    disp(theta_RLSQ);

    % Plot results
    figure;
    plot(Y_test(:, 1), 'b', 'DisplayName', 'Actual');
    hold on;
    plot(1:length(S_sim_LSQ), S_sim_LSQ, 'r', 'DisplayName', 'LSQ Simulated');
    plot(1:length(S_sim_GD), S_sim_GD, 'g', 'DisplayName', 'GD Simulated');
    plot(1:length(S_sim_RLSQ), S_sim_RLSQ, 'b', 'DisplayName', 'RLSQ Simulated');
    legend;
    title(sprintf('Simulation results for k = %d', k));
    xlabel('Time');
    ylabel('S(t)');
    hold off;
end

% Function to simulate the differential equation
function S_sim = simulate_diff_eq(theta, sigma, S0, dt, N)
    S_sim = zeros(N, 1);
    S_sim(1) = S0;
    for t = 2:N
        S_t = S_sim(t-1);
        dS = 0;
        for k = 1:length(theta)-1
            dS = dS + theta(k+1) * S_t^k * dt;
        end
        dz = sqrt(dt) * randn; % Wiener process increment
        dS = dS + sigma * S_t * dz;
        S_sim(t) = S_t + dS;
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