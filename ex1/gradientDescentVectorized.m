function [theta, J_history] = gradientDescentVectorized(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%fprintf("X is %f\n", X);

for iter = 1:num_iters

    for iterInner = 1:m
        % deriv = size(theta)';
        % deriv(1) = (X(iterInner, :) * theta) - y(iterInner);
        % deriv(2) = ((X(iterInner, :) * theta) - y(iterInner)) * X(iterInner, 2);
    
        % theta = theta - alpha/m * deriv;
        theta = theta - alpha/m * [((X(iterInner, :) * theta) - y(iterInner)); (((X(iterInner, :) * theta) - y(iterInner)) * X(iterInner, 2))];
    endfor


    % ============================================================

    % Save the cost J in every iteration
    % Recompute cost f(x)
    J_history(iter) = computeCost(X, y, theta);

    % if (iter < 100)
        fprintf("Hypothesis f(x): %f + %f x\tCost: %f\n", theta(1), theta(2), J_history(iter));
    % endif

end

% fprintf('J_history: %f\n', J_history);

end
