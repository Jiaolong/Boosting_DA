function [alphas best_t] = adaboost( T, labels, hyps )
% [alphas best_t] = adaboost( T, labels, hyps )
%   Train a strong classifier from a set of weak classifiers via adaboost
% Argument
%         T        - maximum iteration times
%         labels   - 1 or -1
%         hyps     - hypothesis of the weak classifiers
% Return
%         strong classifier
%         alphas   - coefficients of the weak classifiers
%         best_t   - best classifier in each iteration
% following the notation: http://en.wikipedia.org/wiki/AdaBoost

alphas = zeros(1,T);
best_t = seros(1,T);
for t=1:T
    % From the family of weak classifiers,
    % find the best weak classifer with minimum error
    min_error = length(labels);
    bestclassifier = 1;
    for i=1:classifiers_size
        predictions = hyps(i, :) == labels';
        error = sum(~predictions.*D);
        if error < min_error
            min_error = error;
            bestclassifier = i;
        end
    end
    
    % stop condition abs(min_error-0.5) < beta
    if min_error > 0.49999 % worse than random guess
        break;
    end
    alphas(t) = 0.5*log((1-min_error)/min_error);
    best_t(t) = bestclassifier;
    % update the weight of each sample
    predictions = hyps(bestclassifier, :) == labels';
    D = D.*exp(2*alphas(t)*(~predictions));
    % normalize so that it is a prob distribution
    Z = sum(D);
    D = D/Z;
    fprintf('\nIter: %d, error = %f, best classifier = %d alpha = %f', t, min_error, bestclassifier, alphas(t));
end

end
