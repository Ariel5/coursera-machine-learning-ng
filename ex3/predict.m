function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

X = [ones(m, 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% Multiply the input stuff by theta1 (first layer), then theta2(2nd layer)
% This should return one of 0-9 digits. NN pretrained by Ng

% After 1st layer. We started w/ 5000x401. Now we're down to 5000x25
l1 = X * Theta1';

% Do the sigmoid. We're inner log-regging
l1 = sigmoid(l1);

% Add [1] col, bcs Theta2 has it. l1 doesn't.
l1 = [ones(size(l1, 1), 1) l1];
% 2nd layer. Now we're down to 5000x10
l2 = l1 * Theta2';

[max_value p] = max(l2, [], 2);

% p(p==10) = 0;


% =========================================================================


end
