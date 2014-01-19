function [hyp_weakclassifiers scoremap] = evaluate_weakclassifiers(samples, labels, model)
% [hyp_weakclassifiers scoremap] = evaluate_weakclassifiers(samples, model)
% test each weak classifiers on training examples
% Argument
%         samples  - training samples, pos and neg
%         model    - training model with weak classifiers
% Return  
%         hyp_weakclassifiers - hypothesis of weak classifiers
%         scoremap            - confidence map

scoremap = zeros(length(model.weakclassifiers), length(samples));
filters = cell(length(model.weakclassifiers),1);
for i=1:length(model.weakclassifiers)
    filters{i} = single(model.weakclassifiers{i}.w);
end
numsamples = length(samples);
numfilters = length(filters);

batchsize = max(1, try_get_matlabpool_size());
inds = 1:numsamples;
for i = 1:batchsize:numsamples
    % do batches of detections in parallel
    thisbatchsize = batchsize - max(0, (i+batchsize-1) - numsamples);
    data = cell(thisbatchsize, 1);
    for k = 1:thisbatchsize
        j = inds(i+k-1);
        fprintf('Computing responses of weak classifiers: %d/%d (%d)\n', i+k-1, numsamples, j);
        resp = fconv(single(samples{j}), filters, 1, numfilters);
        data{k} = [resp{:}]';
    end
    for k = 1:thisbatchsize
        j = inds(i+k-1);
        scoremap(:,j) = data{k};
    end
end

% select threshold
hyp_weakclassifiers = evaluate_thresh(model, scoremap, labels);

end

% select the threshold for weakclassifiers
function hyp_weakclassifiers = evaluate_thresh(model, scoremap, labels)
hyp_weakclassifiers = zeros(size(scoremap));
for i=1:length(model.weakclassifiers)
    score_pos = scoremap(i,:);
    score_pos = score_pos(labels == 1);
    s = sort(score_pos);
    thresh = s(ceil(length(s)*0.25));
    hyp_weakclassifiers(i,:) = (scoremap(i,:) > thresh)*2 -1;%model.weakclassifiers{i}.thresh;
end
end

