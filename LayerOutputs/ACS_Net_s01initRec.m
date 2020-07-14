function s01Init = ACS_Net_s01initRec(image) %输出s01层的初始重构；
tic
%image = imread('./datasets/Test/Submarine.jfif');

%%选择训练的模型路径：
%netfolder = '.\model'; %pretrained model；
%netpaths = dir(fullfile(netfolder,['net',ratio,'.mat']));%pretrained；  
netfolder = 'testmodel';% tianjirong's trained model folder；
netpaths = dir(fullfile(netfolder,['net-epoch-20.mat']));  % switch net path
net = load(fullfile(netfolder,netpaths.name)); 
net = dagnn.DagNN.loadobj(net.net);

%图像转为单精度灰度图；
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

s01Init          = net.vars(net.getVarIndex('s01combine')).value;

toc
end