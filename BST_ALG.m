% Boosting Algorithm
classdef (Sealed) BST_ALG
  properties  (Constant)
    ADABOOST              = 0; % AdaBoost
    TR_ADABOOST           = 1; % Tr_AdaBoost, Boosting for transfer learning
    D_TR_ADABOOST         = 2; % D_Tr_AdaBoost, Boosting for transfer learning using dynamic updating
  end
  methods (Access = private)
    function out = BST_ALG
    end
  end
end

