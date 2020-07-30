%Test Single Scale Reconstruction of CS Net
%This demo evaluate a single scale net ; can be regarded as a standard to compare your model with it;
clear
imageNum = 450;%Choose image No.;
image = imread(['../BigData/datasets/reference-890/',num2str(imageNum),'_img_.png']);
image = rgb2gray(image);
image = ImageNormalize(image);% convert image -> can be mod by 32;
numlayers=7; %choose net model,and means choose Sample Rate meanwhile;
netfolder = '../BigData/model/SubNetIncludeDeep';
netpaths = dir(fullfile(netfolder,['subNet0',num2str(numlayers),'.mat']));  %choose model;
net = load(fullfile(netfolder,netpaths.name)); 
net = dagnn.DagNN.loadobj(net.net);


global useGPU;
if useGPU 
    net.move('gpu');
end
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
net.conserveMemory = false;
net.eval({'input',input});

if useGPU
    initOutput = gather(net.vars(net.getVarIndex(['s0',num2str(numlayers),'combineAdds0',num2str(numlayers-1)])).value);
    deepOutput = gather(net.vars(net.getVarIndex(['s0',num2str(numlayers),'dr_pred'])).value);
else
    initOutput = net.vars(net.getVarIndex(['s0',num2str(numlayers),'combineAdds0',num2str(numlayers-1)])).value;%initial result
    deepOutput = net.vars(net.getVarIndex(['s0',num2str(numlayers),'dr_pred'])).value; %deep result
end

subplot(1,2,1);
imshow(initOutput);
title('Initial Result');
subplot(1,2,2);
imshow(deepOutput);
title('Deep Result');

[psnr_init,ssim_init]        = Cal_PSNRSSIM(im2uint8(input) ,im2uint8(initOutput),0,0);
[psnr_deep,ssim_deep]  = Cal_PSNRSSIM(im2uint8(input) ,im2uint8(deepOutput),0,0);