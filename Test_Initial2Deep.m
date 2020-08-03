tic
clear
netfolder = '../BigData/model/Init2DeepModel/BSDS500_Initial2Deep_bSize64_patch96_stride32';% tianjirong's trained model folder；
netpaths = dir(fullfile(netfolder,'net-epoch-100.mat')); % switch net path
net = load(fullfile(netfolder,netpaths.name)); 
net = dagnn.DagNN.loadobj(net.net);

imageNo = 450;%选择第几张图
InitialRecImage = imread(['../BigData/datasets/reference-890_InitRec/','image',num2str(imageNo),'.png']);
GroundTruth = imread(['../BigData/datasets/reference-890/','image',num2str(imageNo),'.png']);
GroundTruth = modcrop(GroundTruth,32);

%Weather Show or Save  Results ：
showResult  = 1; 
saveResult = 1;
global useGPU;
useGPU =0;
if useGPU == 1
net.move('gpu');
end

%image -> single && Gray；
if size(InitialRecImage,3)==3
    InitialRecImage = rgb2ycbcr(InitialRecImage);
    InitialRecImage = im2single(InitialRecImage(:, :, 1));
else
    InitialRecImage =im2single(InitialRecImage);
end
if size(GroundTruth,3) == 3
    GroundTruth = rgb2ycbcr(GroundTruth);
    GroundTruth = im2single(GroundTruth(:, :, 1));
else
    GroundTruth = im2single(GroundTruth);
end

if useGPU
input = gpuArray(InitialRecImage);
else
input = InitialRecImage;
end
net.conserveMemory = false;
net.eval({'input',input});

if useGPU
        DeepRecImage = gather(net.vars(net.getVarIndex('dr_pred')).value);
else
        DeepRecImage = net.vars(net.getVarIndex('dr_prediction')).value;    %(deep reconstruction prediction);
end

if showResult
    subplot(121),imshow(im2uint8(InitialRecImage));
    title('InitialRecResul');
    subplot(122),imshow(im2uint8(DeepRecImage));
    title('DeepRecResult');
end

%compute PSNR/SSIM etc.
[psnr_init,ssim_init]        = Cal_PSNRSSIM(im2uint8(GroundTruth) ,im2uint8(InitialRecImage),0,0);
[psnr_deep,ssim_deep]  = Cal_PSNRSSIM(im2uint8(GroundTruth) ,im2uint8(DeepRecImage),0,0);
fprintf('PSNR of Initial Reconstruction :%g\n',psnr_init);
fprintf('SSIM of Initial Reconstruction :%g\n',ssim_init);
disp('__________________________________________________');
fprintf('PSNR After Deep Reconstruction :%g\n',psnr_deep);
fprintf('SSIM After Deep Reconstruction :%g\n',ssim_deep);

if saveResult == 1
     imwrite(DeepRecImage,['../BigData/datasets/reference-890_FinalRec/','image',num2str(imageNo),'.png']); 
end
toc