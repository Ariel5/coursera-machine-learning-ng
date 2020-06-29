function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%fprintf("X is %f\n", X);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % If cost increased, go the other way? Cost should never increase.

    sumTheta1 = 0;
    sumTheta2 = 0;

    for iterInner = 1:m
        sumTheta1 += theta(1) + theta(2)*X(iterInner, 2) - y(iterInner);
        sumTheta2 += (theta(1) + theta(2)*X(iterInner, 2) - y(iterInner)) * X(iterInner, 2);
        
    endfor

    sumTheta1 *= 1/m;
    sumTheta2 *= 1/m;


    % First, we need to change thetas
    theta(1) = theta(1) - (alpha * sumTheta1);
    theta(2) = theta(2) - (alpha * sumTheta2);


    % ============================================================

    % Save the cost J in every iteration
    % Recompute cost f(x)
    J_history(iter) = computeCost(X, y, theta);

    fprintf("Hypothesis f(x): %f + %f x\t", theta(1), theta(2));
    fprintf("Cost: %f\n", J_history(iter));

end

end
