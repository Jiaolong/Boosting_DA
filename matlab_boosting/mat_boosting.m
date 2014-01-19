function [alphas best_t] = mat_boosting(Param, labels, domains, hyps)
% [alphas_tar best_t] = boosting(boost_Para, labels, domains, hyp_weakclassifiers)
%   Matlab implementation of boosting algorithms
% Argument
%         boost_Para  - positive and negative samples (features)
%         labels      - 1 or -1
%         domains     - domain label of each sample
%         hyps        - hypothesis of the weal classifiers
% Return
%         strong classifier
%         alphas   - coefficients of the weak classifiers
%         best_t   - best classifier in each iteration

T   = Param.MAX_ITERATION;
alg = Param.NAME_ALGORITHM;
switch alg
    case 0
        fprintf('\nRun Algorithm = %d, AdaBoost',alg);
        [alphas best_t] = adaboost( T, labels, hyps);
    case 1
        fprintf('\nRun Algorithm = %d, TrAdaBoost',alg);
        [alphas best_t] = tr_adaboost( T, labels, domains, hyps);
    case 2
        fprintf('\nRun Algorithm = %d, DTrAdaBoost',alg);
        [alphas best_t] = d_tr_adaboost( T, labels, domains, hyps);
end
end