function net = DeepReconstruction( varargin )
%CNN_DDCN_INIT Summary of this function goes here
%   Detailed explanation goes here
net = dagnn.DagNN();
%net.meta.imdbPath ='D:\下载内容\SCSNet-master\SCSNet-Code\model_64_96_Adam\imdb_BSD500train_patch96_batch64_stride32.mat';
net.meta.imdbPath ='./IMDB/IMDB/SmallTestIMDB.mat'; %选择数据集路径
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
net.meta.trainOpts.numEpochs = 50 ; 
net.meta.adjGradClipping = false;
enl = 1;

net.meta.derOutputs = {'s01pdist',1,'pdistInits01',1,'s02pdist',1,'pdistInits02',1,'s03pdist',1,'pdistInits03',...
1,'s04pdist',1,'pdistInits04',1,'s05pdist',1,'pdistInits05',1,'s06pdist',1,'pdistInits06',1,'s07pdist',1,'pdistInits07',1};
%deep reconstruction
block = dagnn.Conv('size',  [3 3 1 d], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = ['s0' num2str(enl) 'dr1'];
net.addLayer(lName1, block, 'input', lName1, {[lName1 '_f'], [lName1 '_b']});

block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);


i=2;
    block = dagnn.Conv('size',  [1 1 d s], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0'  num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

% net.addLayer([lName1 '_cat'],dagnn.Concat('dim',3),{[lName1 '_relu'],'s01combine'},[lName1 '_cat']);

for i=3:1:15
   
    block = dagnn.Conv('size',  [3 3 s s], 'hasBias', true, ...
                       'stride', 1, 'pad', 1, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0' num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
            
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);
    
%     net.addLayer([lName '_cat'],dagnn.Concat('dim',3),{[lName '_relu'],'s01dr_pred'},[lName '_cat']);

end

i=16;
    block = dagnn.Conv('size',  [1 1 s d], 'hasBias', true, ...
                       'stride', 1, 'pad', 0, 'dilate', 1);
    lName = ['s0' num2str(enl) 'dr' num2str(i)];
    net.addLayer(lName, block, ['s0'  num2str(enl) 'dr' num2str(i-1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName '_relu'],  block, lName, [lName '_relu']);

block = dagnn.Conv('size',  [3 3 d 1], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = ['s0' num2str(enl) 'prediction'];
net.addLayer(lName, block, ['s0'  num2str(enl) 'dr16_relu'], lName, {[lName '_f'], [lName '_b']});

net.addLayer(['s0' num2str(enl) 'dr_pred'],dagnn.Sum(),{['s0' num2str(enl) 'prediction'],['s0' num2str(enl) 'combine']},['s0' num2str(enl) 'dr_pred']);
net.addLayer(['s0' num2str(enl) 'pdist'],dagnn.EuclidLoss(),{['s0' num2str(enl) 'dr_pred'],'label'},['s0' num2str(enl) 'pdist']);
net.initParams();
end