function [alphas_tar best_t] = tr_adaboost( T, labels, domains, hyps, dynamic_cost )
% model = tr_adaboost( samples, labels, domains, model )
%   Train a strong classifier from a set of weak classifiers via Dynamic TrAdaBoost
% Argument
%         T            - maximum iteration times
%         labels       - 1 or -1
%         domains      - domain label of each sample
%         hyps         - hypothesis of the weal classifiers
%         dynamic_cost - if ture, use dynamic cost for auxiliary examples
% Return
%         strong classifier
%         alphas   - coefficients of the weak classifiers
%         best_t   - best classifier in each iteration
% following ICML 2007 Boosting for Transfer Learning

if nargin < 5
    dynamic_cost = false;% dynamic cost
end

alphas_tar = zeros(1,T);
best_t = seros(1,T);
I_aux = find(domains == 0);% indexes of auxiliary samples
I_tar = find(domains == 1);% indexes of target samples
num_aux = length(I_aux);% number of samples in auxiliary domain
for t=1:T
    % Step 1. Probabilities for each sample
    % normalize so that it is a prob distribution
    Z = sum(Wi);
    Wi = Wi/Z;
    
    % Step 2. From the family of weak classifiers,
    % find the best weak classifer with minimum error
    bestclassifier = call_learner(hyps, labels, Wi);
    ht = hyps(bestclassifier, :);
    
    % Step 3. Compute the error on target domain samples
    predictions = (ht(I_tar) == labels(I_tar)');
    error_tar = sum(~predictions.*Wi(I_tar));
    % Stop condition
    if error_tar > 0.49999 % worse than random guess
        break;
    end
    
    % Step 4.
    alphas_tar(t) = 0.5*log((1-error_tar)/error_tar);
    Ct            = 1;
    if dynamic_cost
        Ct = 2*(1 - error_tar);% dynamic cost
    end
    alphs_aux     = Ct*0.5*log(1 + sqrt(2*log(num_aux/T)));
    best_t(t)     = bestclassifier;
    
    % Step 5. Update the weight of each sample
    pre_aux   = (ht(I_aux) == labels(I_aux)');
    Wi(I_aux) = Wi(I_aux).*exp(-alphs_aux*(~pre_aux));
    pre_tar   = (ht(I_tar) == labels(I_tar)');
    Wi(I_tar) = Wi(I_tar).*exp(alphas_tar(t)*(~pre_tar));
    
    fprintf('\nIter: %d, error = %f, best classifier = %d alpha = %f', t, error_tar, bestclassifier, alphas_tar(t));
end
end

% get hyperthesis of weak classifiers and return best weak classifier
function bestclassifier = call_learner(hyp_weakclassifiers, labels, w)
% bestclassifier = call_learner(hyp_weakclassifiers, labels, w)
% Argument 
%          hyp_weakclassifiers - weak classifiers
%          labels              - labels of the samples
%          w                   - weight of the samples
% Return
%          bestclassifier      - the selected weak classifier

min_error = length(labels);
bestclassifier = 1;
for i=1:size(hyp_weakclassifiers,1)
    predictions = hyp_weakclassifiers(i, :) == labels';
    error = sum(~predictions.*w);
    if error < min_error
        min_error = error;
        bestclassifier = i;
    end
end
end
