function net = NKRECNet_init( varargin )
%CNN_DDCN_INIT Summary of this function goes here
%   Detailed explanation goes here
run 'C:\Users\56885\Downloads\matconvnet-1.0-beta25.tar.gz\matconvnet-1.0-beta25\matlab\vl_setupnn.m'
% addpath('C:\Users\56885\Downloads\matconvnet-1.0-beta25.tar.gz\matconvnet-1.0-beta25\matlab')
net = dagnn.DagNN();
% %net.meta.imdbPath ='D:\下载内容\SCSNet-master\SCSNet-Code\model_64_96_Adam\imdb_BSD500train_patch96_batch64_stride32.mat';
% net.meta.imdbPath ='./data/IMDB/BSD500train_TestSet12_batchsize64_imagesize96_stride32.mat'; %选择数据集路径
% net.meta.imdbPath ='C:\codes\matlab\ACSNet-Code\ACSNet-Code\GenerateData\IMDB\SmallTestIMDB.mat'; %选择数据集路径
net.meta.imdbPath ='C:\codes\matlab\ACSNet-Code\ACSNet-Code\GenerateData\twomodel\model1.mat'; %选择非关键帧数据集路径
net.meta.imdbPath1 ='C:\codes\matlab\ACSNet-Code\ACSNet-Code\GenerateData\twomodel\model2.mat'; %选择关键帧数据集路径
d=128;
s=32;
rng('default');
rng(0) ;
reluLeak = 0;
% bnormal =false;
net.meta.solver = 'Adam';
net.meta.inputSize = [96 96] ;
% net.meta.trainOpts.weightDecay = 0.0001 ;
% net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.batchSize = 16; 
% net.meta.trainOpts.batchSize = 32; set batchsize
net.meta.trainOpts.learningRate = [logspace(-3,-3,50) logspace(-4,-4,30) logspace(-5,-5,20)];
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;  %修改epoch
net.meta.trainOpts.numEpochs = 10 ; %%%50

net.meta.adjGradClipping = false;
% net.meta.gradthresh = 0.005;

net.meta.derOutputs = {'pdist',1};


%-------------------------------------------------------------------
%basic layer
% sampling
block = dagnn.Conv('size',  [32 32 1 10], 'hasBias', false, ...
                   'stride', 32, 'pad', [0 0 0 0]);
lName = 'sampling';
net.addLayer(lName, block, 'input', lName, {[lName '_f']});

%initial reconstruction
block = dagnn.Conv('size',  [1 1 10 1024], 'hasBias', false, ...
                   'stride', 1, 'pad', [0 0 0 0]);
lName = 'initRecon';
net.addLayer(lName, block, 'sampling', lName, {[lName '_f']});

block = dagnn.bcs_init_rec_dag('dims',[32 32]);
lName = 'combine';
net.addLayer(lName,block,'initRecon',lName);
% 
% net.addLayer('pdistInits01',dagnn.EuclidLoss(),{lName,'label'},'pdistInits01');%%%%????????
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% % % % % % reconstruction
% % % nkey frame feature extraction
% % % block = dagnn.Conv('size',  {[3 3 1 s], [5 5 1 s], [7 7 1 s]}, 'hasBias', true, ...
% % %                    'stride', 1, 'pad', [1 1 1 1]);
% % % lName = 'nkfeature';
% % % net.addLayer(lName, block, 'combine', lName, {[lName '_f']});

% % key frame feature extraction
% block = dagnn.Conv('size',  {[3 3 1 s], [5 5 1 s], [7 7 1 s]}, 'hasBias', false, ...
%                    'stride', 32, 'pad', [1 1 1 1]);
% % block = dagnn.Conv('size',  {[3 3 1 32], [5 5 1 32], [7 7 1 32]}, ' same');
% lName = 'kfeature';
% % block = dagnn.Conv('size',  {[3 3 1 32], [5 5 1 32], [7 7 1 32]}, 'hasBias', false, ...
% %                    'stride', 32, 'pad', [0 0 0 0]);
% % lName = 'kfeature ';
% net.addLayer(lName, block, 'input', lName, {[lName '_f']});
% 
% %%%连接关键帧非关键帧特征
% net.addLayer('combineadd',dagnn.Concat(),{'nkfeature', 'kfeature'},'combineadd');
% % %---------------------------------------------------------------------------
%%
% % % % % % reconstruction
% % % nkey frame feature extraction
block = dagnn.Conv('size', [3 3 1 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = 'nkfeature1';
net.addLayer(lName, block, 'combine', lName, {[lName '_f'], [lName '_b']});
block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
block = dagnn.Conv('size', [5 5 1 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [2 2 2 2]);
lName = 'nkfeature2';
net.addLayer(lName, block, 'combine', lName, {[lName '_f'], [lName '_b']});
block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
block = dagnn.Conv('size', [7 7 1 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [3 3 3 3]);
lName = 'nkfeature3';
net.addLayer(lName, block, 'combine', lName, {[lName '_f'], [lName '_b']});
block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
net.addLayer('nkcombineadd',dagnn.Concat(),{'nkfeature1_relu', 'nkfeature2_relu', 'nkfeature3_relu'},'nkcombineadd');

% % % key frame feature extraction
block = dagnn.Conv('size', [3 3 1 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = 'kfeature1';
net.addLayer(lName, block, 'input1', lName, {[lName '_f'], [lName '_b']});
block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
block = dagnn.Conv('size', [5 5 1 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [2 2 2 2]);
lName = 'kfeature2';
net.addLayer(lName, block, 'input1', lName, {[lName '_f'], [lName '_b']});
block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
block = dagnn.Conv('size', [7 7 1 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [3 3 3 3]);
lName = 'kfeature3';
net.addLayer(lName, block, 'input1', lName, {[lName '_f'], [lName '_b']});
block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
net.addLayer('kcombineadd',dagnn.Concat(),{'kfeature1_relu', 'kfeature2_relu', 'kfeature3_relu'},'kcombineadd');
% % %%%连接关键帧非关键帧特征
net.addLayer('combineadd',dagnn.Concat(),{'nkcombineadd', 'kcombineadd'},'combineadd');


% % block = dagnn.Conv('size', [3 3 1 s], 'hasBias', true, ...
% %                    'stride', 1, 'pad', [1 1 1 1]);
% % lName = 'kfeature1';
% % net.addLayer(lName, block, 'input', lName, {[lName '_f']});
% % block = dagnn.Conv('size', [5 5 1 s], 'hasBias', true, ...
% %                    'stride', 1, 'pad', [2 2 2 2]);
% % lName = 'kfeature2';
% % net.addLayer(lName, block, 'input', lName, {[lName '_f']});
% % block = dagnn.Conv('size', [7 7 1 s], 'hasBias', true, ...
% %                    'stride', 1, 'pad', [3 3 3 3]);
% % lName = 'kfeature3';
% % net.addLayer(lName, block, 'input', lName, {[lName '_f']});
% % % net.addLayer('kcombineadd',dagnn.Concat(),{'kfeature1', 'kfeature2', 'kfeature3'},'kcombineadd');
% % %%%连接关键帧非关键帧特征
% % net.addLayer('combineadd',dagnn.Concat(),{'nkcombineadd', 'kcombineadd'},'combineadd');
% % 
% % net.addLayer('combineadd',dagnn.Concat(),{'kfeature1', 'kfeature2', 'kfeature3', 'nkfeature1', 'nkfeature2', 'nkfeature3'},'combineadd');



%deep reconstruction
%%%%第一层
i = 1;
block = dagnn.Conv('size',  [3 3 s*3*2 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['dr' num2str(i)];
net.addLayer(lName1, block, 'combineadd', lName1, {[lName1 '_f'], [lName1 '_b']});
% net.addLayer(lName1, block, 'nkcombineadd', lName1, {[lName1 '_f'], [lName1 '_b']});
% net.addLayer(lName1, block, 'nkfeature1_relu', lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);

%%%%第二层
i = 2;
block = dagnn.Conv('size',  [3 3 s s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['dr' num2str(i)];
net.addLayer(lName1, block, ['dr' num2str(i-1) '_relu'], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);
%%%%第三层
i = 3;
net.addLayer(['combine' num2str(i-2) 'add'], dagnn.Concat(),{['dr' num2str(i-2) '_relu'],['dr' num2str(i-1) '_relu']}, ['combine' num2str(i-2) 'add']);

block = dagnn.Conv('size',  [3 3 s*2 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['dr' num2str(i)];
net.addLayer(lName1, block, ['combine' num2str(i-2) 'add'], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);
%%%%第四层
i = 4;
net.addLayer(['combine' num2str(i-2) 'add'], dagnn.Concat(),{['dr' num2str(i-3) '_relu'],['dr' num2str(i-2) '_relu'],['dr' num2str(i-1) '_relu']}, ['combine' num2str(i-2) 'add']);

block = dagnn.Conv('size',  [3 3 s*3 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['dr' num2str(i)];
net.addLayer(lName1, block, ['combine' num2str(i-2) 'add'], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);
%%%%第五层
i = 5;
net.addLayer(['combine' num2str(i-2) 'add'], dagnn.Concat(),{['dr' num2str(i-4) '_relu'], ['dr' num2str(i-3) '_relu'],['dr' num2str(i-2) '_relu'],['dr' num2str(i-1) '_relu']}, ['combine' num2str(i-2) 'add']);

block = dagnn.Conv('size',  [3 3 s*4 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['dr' num2str(i)];
net.addLayer(lName1, block, ['combine' num2str(i-2) 'add'], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);
%%%%第六层
i = 6;
block = dagnn.Conv('size',  [3 3 s 1], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = 'prediction';
net.addLayer(lName1, block, ['dr' num2str(i-1) '_relu'], lName1, {[lName1 '_f'], [lName1 '_b']});

net.addLayer('dr_prediction',dagnn.Sum(),{'prediction','combine'},'dr_prediction');
net.addLayer('pdist',dagnn.EuclidLoss(),{'dr_prediction','label'},'pdist');

net.initParams();