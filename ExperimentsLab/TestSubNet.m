
numSubNet = 7; %Choose Nth SubNets

tic
netfolder = '../BigData/model/SubNetInit';
netpaths = dir(fullfile(netfolder,['SubNetInit0',num2str(numSubNet),'.mat'])); 
net = load(fullfile(netfolder,netpaths.name)); 
net = dagnn.DagNN.loadobj(net.net);

image = imread('../Bigdata/datasets/set12/01.png');
if size(image,3)==3
    image = rgb2ycbcr(image);
    image = im2single(image(:, :, 1));
else
    image =im2single(image);
end
image = modcrop(image,32); 
input = image;
net.conserveMemory = false;
net.eval({'input',input});

% output = net.vars(net.getVarIndex(['s0',num2str(numSubNet),'dr_pred'])).value;
% imshow(output);

s01Init = net.vars(net.getVarIndex(['s0',num2str(numSubNet),'combineAdds0',num2str(numSubNet-1)])).value;
imshow(s01Init);
toc