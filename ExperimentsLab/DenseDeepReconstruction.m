function net = DenseDeepReconstruction( varargin )
%Deep reconstruction net for Initial rec to Deep rec;
net = dagnn.DagNN();
net.meta.imdbPath ='../BigData/IMDB/Initial2Deep_BSDS500_bSize64_patch96_stride32_IMDB.mat'; %Choose Dataset path;
d=128;
s=32;
rng('default');
rng(0);
reluLeak = 0;
% bnormal =false;
net.meta.solver = 'Adam';
net.meta.inputSize = [96 96];
% net.meta.trainOpts.weightDecay = 0.0001;
% net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.batchSize = 64;
% net.meta.trainOpts.batchSize = 32; set batchsize
net.meta.trainOpts.learningRate = [logspace(-3,-3,50) logspace(-4,-4,30) logspace(-5,-5,20)];
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;  %Set training epoch ! ! !
net.meta.trainOpts.numEpochs = 100; 
net.meta.adjGradClipping = false;

%%set output
net.meta.derOutputs = {'pdist',1};

%deep reconstruction
%% Feature extraction 
block = dagnn.Conv('size', [3 3 1 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = 'kfeature1';
net.addLayer(lName, block, 'input', lName, {[lName '_f'], [lName '_b']});
block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
block = dagnn.Conv('size', [5 5 1 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [2 2 2 2]);
lName = 'kfeature2';
net.addLayer(lName, block, 'input', lName, {[lName '_f'], [lName '_b']});
block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
block = dagnn.Conv('size', [7 7 1 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [3 3 3 3]);
lName = 'kfeature3';
net.addLayer(lName, block, 'input', lName, {[lName '_f'], [lName '_b']});
block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
net.addLayer('combineadd',dagnn.Concat(),{'kfeature1_relu', 'kfeature2_relu', 'kfeature3_relu'},'combineadd');

%% Dense net
i = 1;
block = dagnn.Conv('size',  [3 3 s*3 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['dr' num2str(i)];
net.addLayer(lName1, block, 'combineadd', lName1, {[lName1 '_f'], [lName1 '_b']});
% net.addLayer(lName1, block, 'nkcombineadd', lName1, {[lName1 '_f'], [lName1 '_b']});
% net.addLayer(lName1, block, 'nkfeature1_relu', lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);

%%%%第二�?
i = 2;
block = dagnn.Conv('size',  [3 3 s s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['dr' num2str(i)];
net.addLayer(lName1, block, ['dr' num2str(i-1) '_relu'], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);
%%%%第三�?
i = 3;
net.addLayer(['combine' num2str(i-2) 'add'], dagnn.Concat(),{['dr' num2str(i-2) '_relu'],['dr' num2str(i-1) '_relu']}, ['combine' num2str(i-2) 'add']);

block = dagnn.Conv('size',  [3 3 s*2 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['dr' num2str(i)];
net.addLayer(lName1, block, ['combine' num2str(i-2) 'add'], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);
%%%%第四�?
i = 4;
net.addLayer(['combine' num2str(i-2) 'add'], dagnn.Concat(),{['dr' num2str(i-3) '_relu'],['dr' num2str(i-2) '_relu'],['dr' num2str(i-1) '_relu']}, ['combine' num2str(i-2) 'add']);

block = dagnn.Conv('size',  [3 3 s*3 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['dr' num2str(i)];
net.addLayer(lName1, block, ['combine' num2str(i-2) 'add'], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);
%%%%第五�?
i = 5;
net.addLayer(['combine' num2str(i-2) 'add'], dagnn.Concat(),{['dr' num2str(i-4) '_relu'], ['dr' num2str(i-3) '_relu'],['dr' num2str(i-2) '_relu'],['dr' num2str(i-1) '_relu']}, ['combine' num2str(i-2) 'add']);

block = dagnn.Conv('size',  [3 3 s*4 s], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['dr' num2str(i)];
net.addLayer(lName1, block, ['combine' num2str(i-2) 'add'], lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);
%%%%第六�?
i = 6;
block = dagnn.Conv('size',  [3 3 s 1], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = 'prediction';
net.addLayer(lName1, block, ['dr' num2str(i-1) '_relu'], lName1, {[lName1 '_f'], [lName1 '_b']});

net.addLayer('dr_prediction',dagnn.Sum(),{'prediction','input'},'dr_prediction');
net.addLayer('pdist',dagnn.EuclidLoss(),{'dr_prediction','label'},'pdist');
%final output is dr_prediction
%%
net.initParams();
end