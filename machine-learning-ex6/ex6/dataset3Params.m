function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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
%
C_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];


C = C_range(1,1);
sigma = sigma_range(1,1);
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model,Xval);        
Jcv = mean(double(predictions ~= yval));
bestC = C;
bestsigma = sigma;

for C = C_range
    for sigma = sigma_range
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model,Xval);
        Jcv_new = mean(double(predictions ~= yval));
        if (Jcv_new <= Jcv)
            Jcv = Jcv_new;
            bestC = C;
            bestsigma = sigma;
        end
%         visualizeBoundary(X, y, model);
%         fprintf('Program paused. Press enter to continue.\n');
%         pause;
    end
end
fprintf('Best (C,sigma) pair is: (%0.5f, %0.5f)\n', bestC, bestsigma);
fprintf('Jcv is about %0.5f: \n', Jcv);
C = bestC;
sigma = bestsigma;
% =========================================================================

end
