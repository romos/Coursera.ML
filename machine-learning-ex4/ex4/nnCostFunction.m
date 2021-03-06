function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% MY_COMMENT
% transform y into y_binary [0 ... 0 1 0 ... 0]
ybin_template = [1:1:num_labels];
% ybin = zeros(m,num_labels);

% MY_COMMENT
% compute J
A1 = [ones(m, 1) X]';
Z2 = Theta1 * A1;
A2 = sigmoid(Z2);
A2 = [ones(1, m); A2];
Z3 = Theta2 * A2;
A3 = sigmoid(Z3);
h = A3;
% sum i from 1 to m
for i = 1:m,
    h1 = h(:,i); %current output h(x(i)) for the current x(i)
    
    y1 = (ybin_template==y(i,1)); % binary y(i) vector for the current x(i)

    % sum k from 1 to K (using vector multiplication)
    J1 = y1*log(h1) + (1 - y1)*log(1 - h1);
    
    J = J + J1;
end;
J = -1/m * J;

% add regularization terms for Theta1 and Theta2
Theta1_sq = Theta1(:,2:end) .^ 2;
Theta2_sq = Theta2(:,2:end) .^ 2; 
Theta1_sq_sum = sum(Theta1_sq(:));
Theta2_sq_sum = sum(Theta2_sq(:));
J = J + lambda/(2*m) * (Theta1_sq_sum + Theta2_sq_sum);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% MY_COMMENT
% BACKPROPAGATION ALGORITHM:

% templates for partial derivatives. Initializing with zeros.
Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));

% for each training example we need to compute the corresponding deltas
for t = 1:m,
    % set a1 = x1
    a1 = X(t,:)';
    a1 = [1; a1];
    
    % forward propagation to compute a(l) for each network layer
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    % computing delta for the last network layer L
    d_3 = a3 - (ybin_template==y(t,1))';
    
    % computing deltas for other layers backwards
    d_2 = (Theta2')*d_3 .* a2 .* (1 - a2);
    % THE WAY BELOW DOESN'T work because of misleading dimensions.
    % sigmoidGrad(z2) has 25 rows while the 1st part (th*d3) has 26 rows:
    %d_2 = (Theta2')*d_3 .* sigmoidGradient(z2);
    d_2 = d_2(2:end); %skip d_2(0)
    
    % add d_2 for the current example x(t)
    % to the general Delta for the layer l
    Delta_2 = Delta_2 + d_3*(a2');
    Delta_1 = Delta_1 + d_2*(a1');
end;
% divide Deltas by m to obtain the gradients for our cost function
Delta_1 = Delta_1 ./ m;
Delta_2 = Delta_2 ./ m;

Theta1_grad = Delta_1;
Theta2_grad = Delta_2;

% add regularization terms for Theta1 and Theta2. for each l and im but...
% only for j >= 1 (!)
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
