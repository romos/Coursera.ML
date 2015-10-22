function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%% Calculate J(theta):
% % First way to calculate J. Non-vectorized [loop] implementation
% for i = 1:m,
%     thetaXi = X(i,:) * theta;
%     hxi = sigmoid(thetaXi);
%     temp = y(i)*log(hxi) + (1-y(i))*log(1-hxi);
%     J = J + temp;
% end;
% J = -1/m * J;

% Second way to calculate J. Vectorized implementation
Xtheta = X * theta;
hx = sigmoid(Xtheta);
J = y'*log(hx) + (1-y')*log(1-hx);
J = -1/m * J;

%% Calculate Gradient:
% % First way to calculate Gradient. Non-vectorized [loop] implementation
% Xtheta = X * theta;
% hx = sigmoid(Xtheta);
% for j=1:size(theta),
%     for i=1:m,
%         grad(j) = grad(j) + (hx(i)-y(i))*X(i,j);
%     end;
%     grad(j) = 1/m * grad(j);
% end;

% % Second way to calculate Gradient. Vectorized implementation
Xtheta = X * theta;
hx = sigmoid(Xtheta);
grad = 1/m * ( X' * (hx - y) );

% =============================================================
end
