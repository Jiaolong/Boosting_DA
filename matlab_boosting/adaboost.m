function [alphas best_t] = adaboost( T, labels, hyps, Wi )
% Train a strong classifier from a set of weak classifiers via adaboost
% [alphas best_t] = adaboost( T, labels, hyps, Wi )
%
% INPUT
%         T:         maximum iteration times
%         labels:    1 or -1, with shape (num_samplesx1)
%         hyps:      hypothesis of the weak classifiers 
%                    with shape (num_samples x num_classifiers)
%         Wi:        the initial weights, with shape (num_samplesx1)
% OUTPUT
%         strong classifier
%         alphas:   coefficients of the weak classifiers
%         best_t:   best classifier in each iteration
% Reference: http://en.wikipedia.org/wiki/AdaBoost

alphas = zeros(1,T);
best_t = zeros(1,T);
classifiers_size = size(hyps, 2);
for t=1:T
    % From the family of weak classifiers,
    % find the best weak classifer with minimum error
    min_error = length(labels);
    bestclassifier = 1;
    for i=1:classifiers_size
        predictions = hyps(:, i) == labels;
        error = sum(~predictions.*Wi);
        if error < min_error
            min_error = error;
            bestclassifier = i;
        end
    end
    
    % stop condition abs(min_error-0.5) < beta
    if min_error > 0.4999 % worse than random guess
        alphas(t) = 1.0;
        best_t(t) = bestclassifier;
        break;
    end
    alphas(t) = 0.5*log((1-min_error)/min_error);
    best_t(t) = bestclassifier;
    % update the weight of each sample
    predictions = hyps(:, bestclassifier) == labels;
    Wi = Wi.*exp(2*alphas(t)*(~predictions));
    % normalize so that it is a prob distribution
    Z = sum(Wi);
    Wi = Wi/Z;
    fprintf('\nIter: %d, error = %f, best classifier = %d alpha = %f', t, min_error, bestclassifier, alphas(t));
end

end
