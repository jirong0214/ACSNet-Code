%% Experiment with the cnn_mnist_fc_bnorm
% Call 'SCSNet_dag.m' Function to choose network to train !
clc
close all
addpath('matconvnet-1.0-beta25');
[net_bn, info_bn] = SCSNet_dag('expDir', '../BigData/model/Init2DeepModel/BSDS500_Initial2Deep_bSize64_patch96_stride32');