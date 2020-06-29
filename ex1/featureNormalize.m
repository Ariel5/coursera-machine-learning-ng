function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% X(1,:) is first row
%fprintf("X is %f\n", X(1,:));
%fprintf("mean(X[1]) is %f\n", mean(X(1)));
%fprintf("std(X[1]) is %f\n", std(X(1)));


% Mean & std. for each column
mu(1) = mean(X(:,1));
mu(2) = mean(X(:,2));
sigma(1) = std(X(:,1));
sigma(2) = std(X(:,2));

fprintf("mus should be 2000.68, 3.17. They are %f, %f\n", mu(1), mu(2));
fprintf("stds should be 794.7, 0.76. They are %f, %f\n", sigma(1), sigma(2));

for iter = 1:length(X)

	%fprintf("X[1] is %f %f\n", X_norm(iter,1), X_norm(iter,2));

	X_norm(iter,:) -= mu;
	% X_norm(iter,2) -= mu(2);

	% fprintf("After mean, it is %f %f\n", X_norm(iter,1), X_norm(iter,2));

	X_norm(iter, :) /= sigma;
	% X_norm(iter, 2) /= sigma(2);

	% fprintf("At end it is %f %f\n", X_norm(iter,1), X_norm(iter,2));

endfor

%fprintf("After Feature normalization: %f\n", X_norm);


endfunction