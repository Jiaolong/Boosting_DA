function [model, accuracy] = boost_train(conf, Y, D, weak_learners, hyps)
% Train boost classifier
%  [model, accuracy] = boost_train(conf, X, Y, D, weak_learners, hyps)
%
% INPUT
%       conf: configuration
%       Y: training labels, with shape (N x 1)
%       D: domain ids, 0: source, 1: target
%       weak_learners: pre-trained weak learners
%       hyps: pre-computed hypothesis of weak learners
% OUTPUT
%       model: the trained boosted model
%       accuracy: training accuracy

model = struct;
model.algorithm 	= conf.alg_name;
T               	= conf.num_boostIter; % number of iterations

model.weak_learners = weak_learners;

% Initial weight of samples
wi = init_weights(D, conf.mult_target);
% Set the parameter for boosting function
boost_Para = struct(...
    'NAME_ALGORITHM',conf.algorthmId,... % 0: adaboost, 1: tradaboost, 2: d-tradaboost
    'MAX_ITERATION',T);

fprintf('\nStart boosting training ...\n');
th = tic();
if conf.use_mex
    % for mex interface:
    [alphas best_t] = mex_boosting(boost_Para, Y, D, hyps, wi);
    th = toc(th);
    fprintf('\nBoosting mex took %.4f seconds\n', th);
else
    % for matlab interface:
    [alphas best_t] = mat_boosting(boost_Para, Y, D, hyps, wi);
    th = toc(th);
    fprintf('\nBoosting matlab took %.4f seconds\n', th);
end

best_t = best_t(best_t > 0);
alphas = alphas(alphas > 0);
assert(length(alphas) == length(best_t));
assert(~isempty(alphas));

if conf.algorthmId > 0
    ind = min(round(T/2), length(alphas));
    alphas = alphas(ind:end);
    best_t = best_t(ind:end);
end
model.alphas  = alphas;
model.best_t = best_t;

% Evaluate the strong classifier
y_ensemble = hyps(:, best_t) * alphas';
y_ensemble = (y_ensemble > 0)*2 - 1;
accuracy = mean(y_ensemble == Y);
end