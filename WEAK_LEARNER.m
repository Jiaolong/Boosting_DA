% weak random learner type
% can currently train:
% 0. decision stump: look along random dimension of data, choose threshold
% that maximizes information gain in class labels
% 1. 2D linear decision learner: same as decision stump but in 2D. I know,
% in general this could all be folded into single linear stump, but I make
% distinction for historical, cultural, and efficiency reasons.
% 2. Distance learner. Picks a data point in train set and a threshold. The
% label is computed based on distance to the data point
classdef (Sealed) WEAK_LEARNER
  properties  (Constant)
    DECISION_STUMP               = 0; 
    LINEAR_DECISION_2D           = 1;
    DISTANCE                     = 2; 
  end
  methods (Access = private)
    function out = WEAK_LEARNER
    end
  end
end

