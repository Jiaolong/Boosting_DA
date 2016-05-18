function model = weak_train( X, Y, param )
% Weak random learner
%   model = weak_train( X, Y, conf )
%
% INPUT
%      X: training samples of (NXD) shape
%      Y: labels with (Nx1) shape
%      param: parameters of the weak learner
% OUPUT
%      model: the learned model

[N, D] = size(X);
classes = unique(Y);
model = struct;
num_split = param.num_split;
type_classifier = param.type;
model.type_classifier = type_classifier;

max_gain = -1;
switch type_classifier
    case WEAK_LEARNER.DECISION_STUMP
        % proceed to pick optimal splitting value t based on
        % information gain
        for q=1:num_split        
            r = randi(D);
            col = X(:, r);
            tmin = min(col);
            tmax = max(col);
              
            % pick a random threshold
            t = rand(1)*(tmax - tmin) + tmin;
            dec = col < t;
            % evaluate infomation gain
            info_gain = eval_decision(Y, dec, classes);
            if info_gain > max_gain
                max_gain = info_gain;
                model.r = r;
                model.t = t;
            end
        end
    case WEAK_LEARNER.LINEAR_DECISION_2D
        % Linear classifier using 2 dimensions
        % Repeat some number of times:
        % pick two dimensions, pick 3 random parameters, and see what happens
        for q=1:num_split
            r1= randi(D);
            r2= randi(D);
            w= randn(3, 1);
            
            dec = [X(:, [r1 r2]), ones(N, 1)]*w < 0;
            info_gain = eval_decision(Y, dec, classes);
            
            if info_gain > max_gain
                max_gain = info_gain;
                model.r1= r1;
                model.r2= r2;
                model.w= w;
            end
        end
    case WEAK_LEARNER.DISTANCE
        % pick an example and bases decisions on distance threshold
        for q=1:num_split
            x = X(randi(N), :);
            dsts = pdist2(X, x);
            max_dist = max(dsts);
            min_dist = min(dsts);
            
            % randomly select a distance threshold
            t = rand(1)*(max_dist - min_dist) + min_dist;
            dec = dsts < t;
            info_gain = eval_decision(Y, dec, classes);
            
            if info_gain > max_gain
                max_gain = info_gain;
                model.x = x;
                model.t = t;
            end
        end
    otherwise
        fprintf('Weak learner %d not exists', type_classifier);
end

end

function Igain = eval_decision(Y, dec, classes)
% gives Information Gain provided a boolean decision array for what goes
% left or right. classes is unique vector of class labels at this node
YL = Y(dec);
YR = Y(~dec);
H = class_entropy(Y, classes);
HL = class_entropy(YL, classes);
HR = class_entropy(YR, classes);
Igain = H - length(YL)/length(Y)*HL - length(YR)/length(Y)*HR;
end

% Helper function for class entropy used with Decision Stump
function H = class_entropy(y, classes)
cdistr = histc(y, classes) + 1;
cdistr = cdistr/sum(cdistr);
cdistr = cdistr .* log(cdistr);
H = -sum(cdistr);
end

