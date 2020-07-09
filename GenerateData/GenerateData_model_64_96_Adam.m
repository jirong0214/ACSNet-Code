
%%% Generate the training data.

clear;close all;

addpath(genpath('./.'));

batchSize      = 64;        %%% batch size
max_numPatches = batchSize*1400; 
modelName      = 'IMDB';


%%% training and testing
% folder_train  = '\datasets\BSDS500\train';  %%% training
% folder_test   = '\datasets\Set12';    %%% testing

folder_train  = '/Users/tianjirong/OneDrive - stu.ouc.edu.cn/����/ACSNet/ACSNet/ACSNet-Code/datasets/Set12';  %%% training
folder_test   = '/Users/tianjirong/OneDrive - stu.ouc.edu.cn/����/ACSNet/ACSNet/ACSNet-Code/datasets/Set5';    %%% testing

size_input    = 96;          %%% training
size_label    = 96;          %%% testings
stride_train  = 32;          %%% training
stride_test   = 32;          %%% testing
val_train     = 0;           %%% training % default
val_test      = 1;           %%% testing  % default

%%% training patches
[inputs, labels, set]  = patches_generation(size_input,size_label,stride_train,folder_train,val_train,max_numPatches,batchSize);
%%% testing  patches
[inputs2,labels2,set2] = patches_generation(size_input,size_label,stride_test,folder_test,val_test,max_numPatches,batchSize);

inputs   = cat(4,inputs,inputs2);      clear inputs2;
labels   = cat(4,labels,labels2);      clear labels2;
set      = cat(2,set,set2);            clear set2;

if ~exist(modelName,'file')
    mkdir(modelName);
end

%%% save data
save(fullfile(modelName,'SmallTestIMDB'), 'inputs','labels','set','-v7.3')

