function [p_vec, error_train, error_val] = ...
    validationCurveForP(X, y, Xval, yval)
% Try out different p-values and cross-validate to find the best

m = size(X, 1);

p_vec = [1:10];

error_train = zeros(length(p_vec), 1);
error_val = zeros(length(p_vec), 1);

% Find the cost for each p value
% Try out p-values 1-10
for i = 1:length(p_vec)

	% Map X onto Polynomial Features and Normalize
	X_poly = polyFeatures(X, p_vec(i));
	% Not regularize, Normalize! Just brings them closer to between 0 and 1, make them same scale
	% [X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
	X_poly = [ones(m, 1), X_poly];                   % Add Ones

	% Map X_poly_val and normalize (using mu and sigma)
	X_poly_val = polyFeatures(Xval, p_vec(i));
	% X_poly_val = bsxfun(@minus, X_poly_val, mu);
	% X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
	X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

	% Train theta only on training data. Use val set only for finding P
	[theta] = normalEqn(X_poly, y);

	error_train(i) = linearRegCostFunction(X_poly, y, theta, 0);
	error_val(i) = linearRegCostFunction(X_poly_val, yval, theta, 0);
  
endfor

% =========================================================================

end
