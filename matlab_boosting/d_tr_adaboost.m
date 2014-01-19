function [alphas_tar best_t] = d_tr_adaboost( T, labels, domains, hyps )
% model = d_tr_adaboost( samples, labels, domains, model )
%   Train a strong classifier from a set of weak classifiers via Dynamic TrAdaBoost
% Argument
%         T        - maximum iteration times
%         labels   - 1 or -1
%         domains  - domain label of each sample
%         hyps     - hypothesis of the weal classifiers
% Return
%         strong classifier
%         alphas   - coefficients of the weak classifiers
%         best_t   - best classifier in each iteration
% Reference: Adaptive Boosting for Transfer Learning using Dynamic Updates


[alphas_tar best_t] = tr_adaboost( T, labels, domains, hyps, true );
end