function yhat = weak_test(model, X)
% Test weak classifier
% yhat = weakTest(model, X, params)
%
% INPUT
%      model: the trained weak classifier
%      X: test samples, with shape (NxD)
% OUTPUT:
%      yhat: the predicted labels

N = size(X, 1);
type_classifier = model.type_classifier;

switch type_classifier
    case WEAK_LEARNER.DECISION_STUMP
    	% decision stump
    	yhat = double(X(:,model.r) < model.t);
    case WEAK_LEARNER.LINEAR_DECISION_2D
    	% 2-D linear classifier stump
    	yhat = double([X(:, [model.r1 model.r2]) ones(N,1)]*model.w < 0);
    case WEAK_LEARNER.DISTANCE
    	% RBF, distance based classifier
    	yhat = double(pdist2(X, model.x) < model.t);
    otherwise
    	fprintf('Weak learner %d not exists', type_classifier);
end

end