function [Xtrn_src, Ytrn_src, Xtrn_tar, Ytrn_tar, Xtst_tar, Ytst_tar]...
    =load_landmine_data(path_dataset, numLT)
% Load labdmine dataset
% 
%

if nargin < 1
    path_dataset = fullfile('dataset', 'landmine', 'LandmineData.mat');
end

if nargin < 2
    numLT = 20;
end

load(path_dataset)

% source domain data
% Lets consider first five regions as source
% 168 samples with landmine positivies - 5.44%

X_src = []; Y_src = [];

for i = 1:5
    X_src = [X_src; feature{i}];
    Y_src = [Y_src; label{i}];
end
Y_src = Y_src * 2 - 1; % Y \in {-1, 1}

% Target domain data
% Lets consider region 21-25 as target domain
% 161 samples with landmine positives - 7.1683%

X_tar = []; Y_tar = [];

for i = 21:25
    X_tar = [X_tar; feature{i}];
    Y_tar = [Y_tar; label{i}];
end
% shuffle the indices of the target samples
I = randperm(length(Y_tar));
X_tar = X_tar(I,:);
Y_tar = Y_tar(I);
Y_tar = Y_tar * 2 - 1; % Y \in {-1, 1}

Xtrn_src = X_src;
Ytrn_src = Y_src;

idx0 = (Y_tar == -1);
idx1 = ~idx0;
X0 = X_tar(idx0,:);
Y0 = Y_tar(idx0);
X1 = X_tar(idx1,:);
Y1 = Y_tar(idx1);

Xtrn_tar = [X0(1:numLT,:); X1(1:numLT,:)];
Ytrn_tar = [Y0(1:numLT); Y1(1:numLT)];

Xtst_tar = [X0(numLT+1:end,:); X1(numLT+1:end,:)];
Ytst_tar = [Y0(numLT+1:end); Y1(numLT+1:end)];

fprintf('num_src_train_pos: %d\n', sum(Ytrn_src==1));
fprintf('num_src_train_neg: %d\n', sum(Ytrn_src==-1));

fprintf('num_tar_train_pos: %d\n', sum(Ytrn_tar==1));
fprintf('num_tar_train_neg: %d\n', sum(Ytrn_tar==-1));

fprintf('num_tar_test_pos: %d\n', sum(Ytst_tar==1));
fprintf('num_tar_test_neg: %d\n', sum(Ytst_tar==-1));
end