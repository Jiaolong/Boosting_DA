function [alphas best_t] = mat_boosting(Param, labels, domains, hyps, Wi)
% Matlab implementation of boosting algorithms for domain adaptation
% [alphas best_t] = mat_boosting(Param, labels, domains, hyps)
%
% INPUT
%         boost_Para:   positive and negative samples (features)
%         labels:       1 or -1, with shape (num_samplesx1)
%         domains:      domain label of each sample
%         hyps:         hypothesis of the weak classifiers 
%                       with shape (num_samples x num_classifiers)
%         Wi:           initial weights of samples, with shape
%                       (num_samplesx1)
% OUTPUT
%         alphas:       coefficients of the weak classifiers
%         best_t:       best classifier in each iteration

T   = Param.MAX_ITERATION;
alg = Param.NAME_ALGORITHM;
assert(length(labels) == size(hyps,1));
assert(length(Wi) == length(labels));
switch alg
    case BST_ALG.ADABOOST
        fprintf('\nRun Algorithm = %d, AdaBoost',alg);
        [alphas best_t] = adaboost( T, labels, hyps, Wi);
    case BST_ALG.TR_ADABOOST
        fprintf('\nRun Algorithm = %d, TrAdaBoost',alg);
        [alphas best_t] = tr_adaboost( T, labels, domains, hyps, Wi);
    case BST_ALG.D_TR_ADABOOST
        fprintf('\nRun Algorithm = %d, DTrAdaBoost',alg);
        [alphas best_t] = d_tr_adaboost( T, labels, domains, hyps, Wi);
end
end