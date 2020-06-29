function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%fprintf("X is %f\n", X);
%fprintf("theta is %f\n", theta);
%fprintf("y is %f\n", y);

for iter = 1:num_iters

    % We can probably loop through them, but oh well
    sumTheta1 = 0;
    sumTheta2 = 0;
    sumTheta3 = 0;

    for iterInner = 1:m
        sumTheta1 += theta(1) + theta(2)*X(iterInner, 2) + theta(3)*X(iterInner, 3) - y(iterInner)*1; % x^0
        sumTheta2 += (theta(1) + theta(2)*X(iterInner, 2) + theta(3)*X(iterInner, 3) - y(iterInner)) * X(iterInner, 2);
        sumTheta3 += (theta(1) + theta(2)*X(iterInner, 2) + theta(3)*X(iterInner, 3) - y(iterInner)) * X(iterInner, 3);
    endfor

    sumTheta1 *= 1/m;
    sumTheta2 *= 1/m;
    sumTheta3 *= 1/m;


    % First, we need to change thetas
    theta(1) = theta(1) - (alpha * sumTheta1);
    theta(2) = theta(2) - (alpha * sumTheta2);
    theta(3) = theta(3) - (alpha * sumTheta3);

    % ============================================================

    % Save the cost J in every iteration
    % Recompute cost f(x)
    J_history(iter) = computeCostMulti(X, y, theta);

    %fprintf("Hypothesis f(x): %f + %f x + %f x^2\n", theta(1), theta(2), theta(3));
    %fprintf("Cost: %f\n", J_history(iter));

end

end
