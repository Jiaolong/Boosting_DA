function model = boosting( samples, labels, domains, model)
% model = boosting_c( samples, labels, domains, model)
%   Train a strong classifier from a set of weak classifiers via Dynamic TrAdaBoost
% Argument
%         samples  - positive and negative samples (features)
%         labels   - 1 or -1
%         model    - the model
%         domains  - domain label of each sample
% Return
%         model    - a new model with strong classifier
% The C code is 5 times faster than the matlab implementation
% Jiaolong Xu
% jiaolong@cvc.uab.es
% 2013-01-12

addpath('cpp_boosting/');
conf            = config_boosting();
cachedir        = conf.paths.model_dir;
algId           = conf.algorthmId;
model.algorithm = conf.algorithm_names{algId+1};
T               = conf.num_boostIter; % number of iterations

% evaluate weakclassifier on the training samples
try
    load(sprintf([cachedir 'res_weakclassifiers_%s.mat'],model.targetset));
catch
    [hyp_weakclassifiers scoremap] = evaluate_weakclassifiers(samples, labels, model);
    save(sprintf([cachedir 'res_weakclassifiers_%s.mat'],model.targetset), 'hyp_weakclassifiers', 'scoremap');
end

try
    load(sprintf([cachedir 'model_%s_%s_c.mat'],model.algorithm, model.targetset));
catch
    % initial weight of samples
    wi = compute_weight_init(samples, labels, domains);
    % Set the parameter for boosting function
	boost_Para = struct(...
    'NAME_ALGORITHM',algId,... % 0: adaboost, 1: tradaboost, 2: d-tradaboost
    'MAX_ITERATION',T);
    
    th = tic();
    [alphas_tar best_t] = mex_boosting(boost_Para, labels, domains, hyp_weakclassifiers', wi);
    th = toc(th);
    fprintf('\nBoosting took %.4f seconds', th);
    
    model.alpha  = alphas_tar;
    model.best_t = best_t;
    % backup the original holistic model
    model.w0 = model.w;
    
    % update the strong classifier
    w1 = zeros(size(model.w));
    for i=1:T
        ind = best_t(i);
        w1 = w1 + model.alpha(i).*model.weakclassifiers{ind}.w;
    end
    w1 = w1./(norm(w1(:))+eps);
    
    % average of the selected classifiers
    w2 = zeros(size(model.w));
    for i=1:length(best_t)
        ind = best_t(i);
        w2 = w2 + model.weakclassifiers{ind}.w;
    end
    w2 = w2./(norm(w2(:))+eps);
    
    model.w1 = w1;
    model.w2 = w2;

    save(sprintf([cachedir 'model_%s_%s_c.mat'],model.algorithm, model.targetset), 'model');
end
end

function wi = compute_weight_init(samples, labels, domains)
% compute the initial sample weight
I = labels == 1; % positives
samples_p = samples(I);
domains_p = domains(I);

aux_samples = samples_p(domains_p == 0);
tar_samples = samples_p(domains_p == 1);

aux_dis = zeros(size(aux_samples,1),1);% nearest distance to the target
parfor i=1:size(aux_samples,1)
    aux = aux_samples{i};
    s1 = aux(:);
    min_dis = bitmax;% initial dis
    for j=1:size(tar_samples,1)
        tar = tar_samples{j};
        s2  = tar(:);
        dis = norm(s1-s2);
        if dis < min_dis
            min_dis = dis;
        end
    end
    aux_dis(i) = min_dis;
end
wi_aux = aux_dis./sum(aux_dis,1);% 1-norm
wi = ones(size(samples,1),1);
n = 1;
for i=1:length(wi)
    if (labels(i) == 1)&& (domains(i)==0)
        wi(i) = wi_aux(n);
        n = n + 1;
    end
end
end
