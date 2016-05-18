function accuracy = boost_test(model, X, Y)
% Test boost classifier
%  model = boost_train(X, Y, conf)
%
% INPUT
%       model:      the pre-trained boosted model
%       X:          training samples, with shape (N x D)
%       Y:          training labels, with shape (N x 1)
% OUTPUT
%       accuracy:   the testing accuracy

num_weak_learners 	= length(model.weak_learners);
N = size(X, 1);
% Evaluate weak classifiers
hyps = zeros(N, num_weak_learners);

for i=1:num_weak_learners
    yhat = weak_test(model.weak_learners{i}, X);
    yhat = (yhat > 0)*2 - 1;
    hyps(:, i) = yhat;
end

% Evaluate the strong classifier
y_ensemble = hyps(:, model.best_t) * model.alphas';
y_ensemble = (y_ensemble > 0)*2 - 1;
accuracy = mean(y_ensemble == Y);
end