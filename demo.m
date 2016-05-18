%--------------------------------------------------
% Demo of Adaboost
% Copyright (c) 2016
% Written by Jiaolong Xu
%--------------------------------------------------

% We reset the RNG's seed to a fixed value so that experimental results 
% are reproducible.
stream = RandStream('mt19937ar','Seed',1000);
RandStream.setGlobalStream(stream);

% load configuration
conf = config_boosting();

% Load training data
path_dataset = fullfile('dataset', 'landmine', 'LandmineData.mat');
numLT = 15;
[Xtrn_src, Ytrn_src, Xtrn_tar, Ytrn_tar, Xtst_tar, Ytst_tar] =...
    load_landmine_data(path_dataset, numLT);

Xtrn = [Xtrn_src; Xtrn_tar];
Ytrn = [Ytrn_src; Ytrn_tar];
num_samples    = size(Xtrn, 1);
domains        = zeros(num_samples, 1);
domains(1+length(Ytrn_src):end) = 1;

% Get weak learners
cache_file = fullfile(conf.cache_dir, 'weaklearners.mat');
try
    load(cache_file);
catch
    [weak_learners, hyps] = get_weak_learners(Xtrn, Ytrn);
    save(cache_file, 'weak_learners', 'hyps');
end

acc_algs = zeros(3,2);
%-------------------------------------------------------------
% Train adaboost
conf.algorthmId                    = BST_ALG.ADABOOST;% boosting algorithm. 0: adaboost, 1: tradaboost, 2: dtradaboost
conf.alg_name 					   = conf.algorithm_names{conf.algorthmId + 1};

[model, accuracy] = boost_train(conf, Ytrn, domains, weak_learners, hyps);
fprintf('\n[%s]: Training accuracy: %f\n', conf.alg_name, accuracy);
acc_algs(1,1) = accuracy;

% Test adaboost
accuracy = boost_test(model, Xtst_tar, Ytst_tar);
fprintf('[%s]: Testing accuracy: %f\n', conf.alg_name, accuracy);
acc_algs(1,2) = accuracy;

%-------------------------------------------------------------
% Train tr_adaboost
conf.algorthmId                    = BST_ALG.TR_ADABOOST;% boosting algorithm. 0: adaboost, 1: tradaboost, 2: dtradaboost
conf.alg_name 					   = conf.algorithm_names{conf.algorthmId + 1};
[model, accuracy] = boost_train(conf, Ytrn, domains, weak_learners, hyps);
fprintf('\n[%s]: Training accuracy: %f\n', conf.alg_name, accuracy);
acc_algs(2,1) = accuracy;

% Test tr_adaboost
accuracy = boost_test(model, Xtst_tar, Ytst_tar);
fprintf('[%s]: Testing accuracy: %f\n', conf.alg_name, accuracy);
acc_algs(2,2) = accuracy;

%-------------------------------------------------------------
% Train d_tr_adaboost
conf.algorthmId                    = BST_ALG.D_TR_ADABOOST;% boosting algorithm. 0: adaboost, 1: tradaboost, 2: dtradaboost
conf.alg_name 					   = conf.algorithm_names{conf.algorthmId + 1};
[model, accuracy] = boost_train(conf, Ytrn, domains, weak_learners, hyps);
fprintf('\n[%s]: Training accuracy: %f\n', conf.alg_name, accuracy);
acc_algs(3,1) = accuracy;

% Test d_tr_adaboost
accuracy = boost_test(model, Xtst_tar, Ytst_tar);
fprintf('[%s]: Testing accuracy: %f\n\n', conf.alg_name, accuracy);
acc_algs(3,2) = accuracy;

fprintf('Testing accuracy on target domain:\n');
for i=1:3
    fprintf('[%s]:\t %f\n',...
        conf.algorithm_names{i}, acc_algs(i, 2));
end