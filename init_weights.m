function wi = init_weights(domains, mult_target)
% Compute the initial sample weight
% wi = init_weights(domains)
%
% INPUT
%       domains: domain ids (0: source, 1: target)
%       target_weight: weight multiply on target samples
% OUTPUT
%       wi: initial weight of each sample

if nargin < 2
    mult_target = 10;
end
wi = ones(length(domains),1);
wi(domains == 1) = mult_target;
wi = wi/sum(wi);
end