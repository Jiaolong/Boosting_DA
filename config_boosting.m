function conf = config_boosting()
% Set up configuration variables

conf.algorithm_names               = {'adaboost' 'tradaboost' 'dtradaboost'};
conf.algorthmId                    = BST_ALG.D_TR_ADABOOST;% boosting algorithm. 0: adaboost, 1: tradaboost, 2: dtradaboost

if ispc
    conf.paths.base_dir = 'cache\trboosting\';
    conf.paths.model_dir = 'cache\trboosting\';
else
    conf.paths.base_dir = 'cache/trboosting/';
    conf.paths.model_dir = 'cache/trboosting/';
end
if ~exist(conf.paths.base_dir, 'dir')
    mkdir(conf.paths.base_dir);
end
if ~exist(conf.paths.model_dir, 'dir')
    mkdir(conf.paths.model_dir);
end
end

