
%%% Generate the training data.

clear;close all;

addpath(genpath('./.'));

batchSize      = 64;        %%% batch size
max_numPatches = batchSize*1400; 
modelName      = '../BigData/IMDB';


%%% training and testing
% folder_train  = '\datasets\BSDS500\train';  %%% training
% folder_test   = '\datasets\Set12';    %%% testing

folder_train_initialRec      = '../BigData/datasets/DataSetForInitial2Deep/initialRecforTrain';  %%% training dataset : initial rec;
folder_train_Originimage = '../BigData/datasets/DataSetForInitial2Deep/originImageforTrain';   %%% training dataset :origin image;
folder_test_initialRec       = '../BigData/datasets/DataSetForInitial2Deep/initialRecforTest';    %%% testing dataset : initial rec;
folder_test_Originimage  = '../BigData/datasets/DataSetForInitial2Deep/originImageforTest';    %%% testing  dataset :origin image;

size_input    = 96;          %%% training
size_label    = 96;          %%% testings
stride_train  = 32;          %%% training
stride_test   = 32;          %%% testing
val_train     = 0;           %%% training % default
val_test      = 1;           %%% testing  % default

%%% training patches
[inputs, Invalidlabels, Invalidset]  = patches_generation(size_input,size_label,stride_train,folder_train_initialRec,val_train,max_numPatches,batchSize);
[Invalidinputs, labels, set]  = patches_generation(size_input,size_label,stride_train,folder_train_Originimage,val_train,max_numPatches,batchSize);

%%% testing  patches
[inputs2,Invalidlabels2,Invalidset2] = patches_generation(size_input,size_label,stride_test,folder_test_initialRec,val_test,max_numPatches,batchSize);
[Invalidinputs2,labels2,set2] = patches_generation(size_input,size_label,stride_test,folder_test_Originimage,val_test,max_numPatches,batchSize);


inputs   = cat(4,inputs,inputs2);      clear inputs2;
labels   = cat(4,labels,labels2);      clear labels2;
set      = cat(2,set,set2);            clear set2;

% if ~exist(modelName,'file')
%     mkdir(modelName);
% end

%%% save data
save(fullfile(modelName,'Initial2Deep_BSDS500_bSize64_patch96_stride32_IMDB'), 'inputs','labels','set','-v7.3')

