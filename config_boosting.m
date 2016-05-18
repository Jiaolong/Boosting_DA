function conf = config_boosting()
% Set up configuration variables

conf.algorithm_names               = {'adaboost' 'tradaboost' 'dtradaboost'};
conf.algorthmId                    = BST_ALG.ADABOOST;% boosting algorithm. 0: adaboost, 1: tradaboost, 2: dtradaboost
conf.alg_name 					   = conf.algorithm_names{conf.algorthmId + 1};
conf.num_boostIter                 = 400; % boost iteration times

conf.weak_learner.type             = WEAK_LEARNER.LINEAR_DECISION_2D; % weak classifier
conf.weak_learner.num_split        = 5; % number of splits
conf.num_weak_learners             = 400; % number of weak learners
conf.mult_target                   = 1; % ratio of the target and source intial weights
conf.use_mex                       = false;
if conf.use_mex
    addpath('cpp_boosting/'); % use mex implementation
else
    addpath('matlab_boosting/'); % use matlab implementation
end

conf.cache_dir                     = 'cache';
if ~exist(conf.cache_dir, 'dir')
    mkdir(conf.cache_dir);
end
end

