tic
netfolder = '../BigData/testmodel_batchsize64';% tianjirong's trained model folder；
netpaths = dir(fullfile(netfolder,['net-epoch-98.mat'])); % switch net path
net = load(fullfile(netfolder,netpaths.name)); 
net = dagnn.DagNN.loadobj(net.net);
image = imread('/Users/tianjirong/OneDrive - stu.ouc.edu.cn/代码/ACSNet/ACSNet-Code/RecResult/策略2/FIsh/base005/TotalinitRec.jpg');
useGPU      = 0;  
if useGPU
net.move('gpu');
end
%选择是否要显示和保存图像重构结果：
showResult  = 1; 
SaveImage = 0;
%图像转为单精度灰度图；
if size(image,3)==3
image = rgb2ycbcr(image);
image = im2single(image(:, :, 1));
else
image =im2single(image);
end
image = modcrop(image,32); 
if useGPU
input = gpuArray(image);
else
input = image;
end
label = image;
net.conserveMemory = false;
net.eval({'input',input});

if useGPU
        output = gather(net.vars(net.getVarIndex('s01dr_pred')).value);
else
        %output = net.vars(net.getVarIndex('s01dr_pred')).value;    %(deep reconstruction prediction);
        output = net.vars(net.getVarIndex('s01dr_pred')).value; 
end

if showResult
    figure;
    imshow(im2uint8(output)) ;
    title('DeepRecResult');
%     figure;
%     imshow(im2uint8(image));  %sr = 0.01finalout + sr = 0.5residual;
%     title('OriginalImage');
end
%保存图像
%imwrite(s07initOutput,'./RecResult/Submarine_s07initResidual.jpg'); 
toc