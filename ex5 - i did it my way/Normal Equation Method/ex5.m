clear ; close all; clc

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
% This sets up the data automatically?? How?
load ('ex5data1.mat');

% % Join all for best normal eq. fit
% X = [X;Xval];
% y = [y;yval];

% m = Number of examples
m = size(X, 1);

% Linear function fit
% [theta] = normalEqn([ones(m, 1) X], y);

% h = [ones(size(Xtest, 1), 1) Xtest] * theta;
% [h, ytest]

% % Plot test data and fit
% figure(1);
% plot(Xtest, ytest, 'rx', Xtest, h, 'go', 'MarkerSize', 10, 'LineWidth', 1.5);
% xlabel('Change in water level (x_val)');
% ylabel('Water flowing out of the dam (y_val)');
% % axis([-20 50 -20 80]);
% title (sprintf('Linear Function Regression Fit'));

% test_cost = linearRegCostFunction(Xtest, ytest, theta, 0)

% Find the best p-value (nr of polynomials to use)
% [p_vec, error_train, error_val] = validationCurveForP(X, y, Xval, yval);

% close all;
% plot(p_vec, error_train, p_vec, error_val);
% legend('Train', 'Cross Validation');
% xlabel('p');
% ylabel('Error');

% fprintf('p\t\tTrain Error\tValidation Error\n');
% for i = 1:length(p_vec)
% 	fprintf(' %f\t%f\t%f\n', ...
%             p_vec(i), error_train(i), error_val(i));
% end

% Based on cross-val error, best p = 3
p = 3;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
% Not regularize, Normalize! Just brings them closer to between 0 and 1, make them same scale
% [X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
% X_poly_test = bsxfun(@minus, X_poly_test, mu);
% X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
% X_poly_val = bsxfun(@minus, X_poly_val, mu);
% X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

[theta] = normalEqn(X_poly, y);


% Test on cross-val set to find the best p (without looking at test set)
hval = X_poly_val * theta;

theta

fprintf("Hypothesis_val, yval = ");
[hval, yval]

close all;

% Plot test data and fit
% figure(1);
% plot(Xval, yval, 'rx', Xval, hval, 'go', 'MarkerSize', 10, 'LineWidth', 1.5);
% xlabel('Change in water level (x_val)');
% ylabel('Water flowing out of the dam (y_val)');
% % axis([-20 50 -20 80]);
% title (sprintf('Polynomial Function Regression Fit'));

val_cost = linearRegCostFunction(X_poly_val, yval, theta, 0)


% % Test on cross-val set to find the best p (without looking at test set)
htest = X_poly_test * theta;
[htest, ytest]

close all;

% Plot test data and fit
figure(1);
plot(Xtest, ytest, 'rx', Xtest, htest, 'go', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x_val)');
ylabel('Water flowing out of the dam (y_val)');
% axis([-20 50 -20 80]);
title (sprintf('Polynomial Function Regression Fit'));

test_cost = linearRegCostFunction(X_poly_test, ytest, theta, 0)