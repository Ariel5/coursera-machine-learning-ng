function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_vec = [1,5,10,25,50,75,100];
sigma_vec = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

fprintf('Determining best C and sigma ...\n')

% values = zeros(length(C_vec), length(sigma_vec));

% for i=1:length(C_vec)
% 	each_sigma_for_C = (zeros(length(sigma_vec), 1))';
% 	for j=1:length(sigma_vec)
% 		C = C_vec(i);
% 		sigma = sigma_vec(j);
% 		% Train on training, cross-val C and sigma on cval data
% 		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
% 		predictions = svmPredict(model, Xval);
% 		each_sigma_for_C(j) = mean(double(predictions ~= yval));
% 	endfor
% 	values(i, :) = each_sigma_for_C;
% endfor

% % Save values output to file, so we don't run this 15-min operation every time
% save C_and_sigma_values.mat values;

% Seems from all the testing above, this has the lowest error (.03)
C = 1;
sigma = 0.1;

% =========================================================================

end
