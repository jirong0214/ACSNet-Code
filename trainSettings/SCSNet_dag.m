function [net, info] = SCSNet_dag(varargin)
opts.batchNormalization = false ;
opts.networkType = 'dagnn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
% if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(vl_rootnn, 'data', ['dcCSNetRes-' sfx]) ;
%opts.expDir = '../BigData/model';
[opts, varargin] = vl_argparse(opts, varargin) ;

% opts.dataDir = fullfile(vl_rootnn, 'data', 'mnist') ;
% opts.imdbPath = fullfile(opts.expDir, 'data_x3_SR_64x64_raw.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end %GPU Settings! ! !
% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------
%net = DeepReconstruction('networkType', opts.networkType) ;%Choose which network to train
net = DenseDeepReconstruction('networkType', opts.networkType);
imdb = load(net.meta.imdbPath) ;
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
switch opts.networkType
  case 'simplenn', trainfn = @cnn_rsCSNetRes_train ;
  case 'dagnn', trainfn = @SCSNet_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', imdb.set == 2,...
  'derOutputs',net.meta.derOutputs);

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y);
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
% labels = imdb.images.labels(1,batch) ;
labels = imdb.images.labels(:,:,:,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.inputs(:,:,:,batch) ;
labels = imdb.labels(:,:,:,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
  labels = gpuArray(labels);
end
inputs = {'input', images, 'label', labels} ;