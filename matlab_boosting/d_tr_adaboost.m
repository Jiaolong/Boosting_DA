function [alphas_tar best_t] = d_tr_adaboost( T, labels, domains, hyps, Wi)
% Train a strong classifier from a set of weak classifiers via 
% Dynamic TrAdaBoost
% [alphas_tar best_t] = d_tr_adaboost( T, labels, domains, hyps, Wi)
%
% INPUT
%         T:         maximum iteration times
%         labels:    1 or -1, (N x 1)
%         domains:   domain label of each sample
%         hyps:      hypothesis of the weal classifiers,
%                    (N x num_classifiers)
%         Wi:        initial weights of the samples (N x 1)
% OUTPUT
%         strong classifier
%         alphas:    coefficients of the weak classifiers
%         best_t:    best classifier in each iteration
% Reference: 
%S. Al-Stouhi and C. Reddy. Adaptive boosting for transfer learning using 
% dynamic updates. In ECML PKDD, Athens, Greece, 2011.

[alphas_tar best_t] = tr_adaboost( T, labels, domains, hyps, Wi, true );
end