%% Experiment with the cnn_mnist_fc_bnorm
clc
close all

% [net_bn, info_bn] = SCSNet_dag(...
%   'expDir', './TrainedModel');
addpath('BigData');
[net_bn, info_bn] = SCSNet_dag(...
  'expDir', '../BigData/testmodel_batchsize32');
% Call 'SCSNet_dag.m' Function to choose network to train£º
% case 1.simplenn -->cnn_rsCSNetRes_train     
% case 2.dagnn     --> SCSNet_train_dag    % we choose this one;