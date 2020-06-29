function J = logRegComputeCostRegularized(h, z, y, m, Theta1, Theta2, lambda)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
% m = length(h); % number of training examples

% You need to return the following variables correctly 
J = 0;
% deriv (cost)
% grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the particular
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta

% h = sigmoid(X*theta);

J = -1/m * (y'*log(h) + (1-y)'*log(1-h));

%gradient would be the derivative of the cost f(x)

% Sum of diagonal nrs. This is because y is a matrix, not a vector. y .* h should be elementwise product
J = trace(J);

% % exempting theta0
J += lambda/(2*m) * (sum(sum(Theta1'(2:length(Theta1'(:, 1)), :).^2)) + sum(sum(Theta2'(2:length(Theta2'(:, 1)), :).^2)));
% grad += lambda/m * [0; theta(2:length(theta))];



end
