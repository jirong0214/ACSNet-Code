netfolder = '../BigData/model/SubNetIncludeDeep';
netpaths = dir(fullfile(netfolder,['subNet07.mat'])); 
net = load(fullfile(netfolder,netpaths.name)); 
net = dagnn.DagNN.loadobj(net.net);