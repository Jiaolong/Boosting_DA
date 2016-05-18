function [weak_learners, hyps] = get_weak_learners(X, Y)
% Train and evaluate weak classifiers

conf            	= config_boosting();
num_weak_learners 	= conf.num_weak_learners;
N                   = size(X, 1);
hyps = zeros(N, num_weak_learners);
count = 1;
weak_learners = cell(num_weak_learners, 1);
fprintf('Training weak learners ...\n');
while true
    if count > num_weak_learners
        break;
    end
    weak_clf = weak_train(X, Y, conf.weak_learner);
    yhat = weak_test(weak_clf, X);
    yhat = (yhat > 0)*2 - 1;
    acc = mean(yhat==Y);
    if acc >= 0.5
        hyps(:,count) = yhat;
        weak_learners{count} = weak_clf;
        count = count + 1;
        if mod(count, 50) == 0
            fprintf('[%d/%d]\n', count, num_weak_learners);
        end
    end
end
weak_learners = weak_learners;
end