%% Experiment with the cnn_mnist_fc_bnorm
clc
close all

% [net_bn, info_bn] = SCSNet_dag(...
%   'expDir', './TrainedModel');
addpath('BigData');
[net_bn, info_bn] = SCSNet_dag(...
  'expDir', 'testmodel');
% 调用SCSNet_dag函数可选择网络类型：
% case 1.simplenn -->cnn_rsCSNetRes_train     
% case 2.dagnn     --> SCSNet_train_dag    %我们选用的这个。