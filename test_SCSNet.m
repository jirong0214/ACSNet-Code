% clear all; close all;

addpath('D:\下载内容\SCSNet-master\SCSNet-Code\data\utilities');
run('D:\Matlab\MatConvNet\matconvnet-1.0-beta25\matlab\vl_setupnn.m') ;
tic
ratio = '01'; % coarse granular corresponds to the sampling ratons of 0.01 0.05 0.1 0.2 0.3 0.4 and 0.5 in the pretrained model.
num_epoch = 100;%switch epoch

%netfolder = '.\model'; %pretrained model
netfolder = '.\results';% tianjirong's trained model folder；
res01 = zeros(200,2);
res02 = zeros(200,2);
res03 = zeros(200,2);
res04 = zeros(200,2);
res05 = zeros(200,2);
res06 = zeros(200,2);
res07 = zeros(200,2);

%netpaths = dir(fullfile(netfolder,['net',ratio,'.mat']));%pretrained；  
netpaths = dir(fullfile(netfolder,['net-epoch-100.mat']));  % switch net path
net = load(fullfile(netfolder,netpaths.name)); 
net = dagnn.DagNN.loadobj(net.net);
 
showResult  = 1; % need show result when test?
useGPU      = 1;  
pauseTime   = 1;
SaveImage = 1;
if useGPU
%         net1.move('gpu');
    net.move('gpu');
end
 
folderTest = 'C:\Users\Melbourne\OneDrive - stu.ouc.edu.cn\代码\ACSNet\ACSNet\ACSNet-Code\blockimage\baby_GT.bmp';%  switch test folder
ext         =  {'*.jpg','*.png','*.bmp','*.tif'};
filepaths   =  [];
save_dir = ['C:\Users\Melbourne\OneDrive - stu.ouc.edu.cn\代码\ACSNet\ACSNet\ACSNet-Code\blockimage\baby_GT.bmp_Rec\']
if exist(save_dir)==0
    mkdir(fullfile(save_dir));
end
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(folderTest,ext{i})));
end


PSNRs01 = zeros(1,length(filepaths));
SSIMs01 = zeros(1,length(filepaths));
PSNRs02 = zeros(1,length(filepaths));
SSIMs02 = zeros(1,length(filepaths));
PSNRs03 = zeros(1,length(filepaths));
SSIMs03 = zeros(1,length(filepaths));
PSNRs04 = zeros(1,length(filepaths));
SSIMs04 = zeros(1,length(filepaths));
PSNRs05 = zeros(1,length(filepaths));
SSIMs05 = zeros(1,length(filepaths));
PSNRs06 = zeros(1,length(filepaths));
SSIMs06 = zeros(1,length(filepaths));
PSNRs07 = zeros(1,length(filepaths));
SSIMs07 = zeros(1,length(filepaths));

for i = 1:length(filepaths)
    image = imread(['C:\Users\Melbourne\OneDrive - stu.ouc.edu.cn\代码\ACSNet\ACSNet\ACSNet-Code\blockimage\baby_GT.bmp\','m',num2str(i),'_baby_GT.bmp']);
% image = imread(fullfile(folderTest,filepaths(i).name));
    i
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
    %net.conserveMemory = false;
    net.conserveMemory = true;
    net.eval({'input',input});

    if useGPU
          %output01 = gather(net.vars(net.getVarIndex('s01dr_pred')).value);% + gather(net.vars(net.getVarIndex('s02dr_pred')).value);
%          output02 = gather(net.vars(net.getVarIndex('s02dr_pred')).value);
%          output03 = gather(net.vars(net.getVarIndex('s03dr_pred')).value);%此 MATLAB 函数 对尚未计算、但需要计算的 tall 数组 X 执行所有必需的排队运算，然后将结果作为 Y 收集到内存中。
%          output04 = gather(net.vars(net.getVarIndex('s04dr_pred')).value);
%          output05 = gather(net.vars(net.getVarIndex('s05dr_pred')).value);
%          output06 = gather(net.vars(net.getVarIndex('s06dr_pred')).value);
             output07 = gather(net.vars(net.getVarIndex('s07dr_pred')).value);
    else
         output01 = net.vars(net.getVarIndex('s01dr_pred')).value;
         output02 = net.vars(net.getVarIndex('s02dr_pred')).value;
         output03 = net.vars(net.getVarIndex('s03dr_pred')).value;
         output04 = net.vars(net.getVarIndex('s04dr_pred')).value;
         output05 = net.vars(net.getVarIndex('s05dr_pred')).value;
         output06 = net.vars(net.getVarIndex('s06dr_pred')).value;
         output07 = net.vars(net.getVarIndex('s07dr_pred')).value;
    end
    
     [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output01),im2uint8(label),0,0);
     PSNRs01(i) = PSNRCur;
     SSIMs01(i) = SSIMCur; 
%      [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output02),im2uint8(label),0,0);
%      PSNRs02(i) = PSNRCur;
%      SSIMs02(i) = SSIMCur; 
%      [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output03),im2uint8(label),0,0);
%      PSNRs03(i) = PSNRCur;
%      SSIMs03(i) = SSIMCur; 
%      [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output04),im2uint8(label),0,0);
%      PSNRs04(i) = PSNRCur;
%      SSIMs04(i) = SSIMCur; 
%      [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output05),im2uint8(label),0,0);
%      PSNRs05(i) = PSNRCur;
%      SSIMs05(i) = SSIMCur; 
%      [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output06),im2uint8(label),0,0);
%      PSNRs06(i) = PSNRCur;
%      SSIMs06(i) = SSIMCur; 
      [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(output07),im2uint8(label),0,0);
      PSNRs07(i) = PSNRCur;
      SSIMs07(i) = SSIMCur; 
toc
%    if showResult          %显示和原图对比的图像
%         imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output03)));
%         title([filepaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
%         drawnow;
%         pause(pauseTime)
%    end 
        if showResult
            imshow(im2uint8(output07))
            title(['m',num2str(i),'_baby_GT.bmp','    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
            drawnow;
            %pause(pauseTime)
        end
        if SaveImage
             imwrite(output07,strcat(save_dir,['m',num2str(i),'_baby_GT.bmp']));  
        end
% res01(iii,:) = [mean(PSNRs01),mean(SSIMs01)];
% disp([mean(PSNRs01),mean(SSIMs01)])
% res02(iii,:) = [mean(PSNRs02),mean(SSIMs02)];
% disp([mean(PSNRs02),mean(SSIMs02)])
%res03(iii,:) = [mean(PSNRs03),mean(SSIMs03)];%---------------------------
%disp([mean(PSNRs03),mean(SSIMs03)])
% res04(iii,:) = [mean(PSNRs04),mean(SSIMs04)];
%disp([mean(PSNRs04),mean(SSIMs04)])
% res05(iii,:) = [mean(PSNRs05),mean(SSIMs05)];
% disp([mean(PSNRs05),mean(SSIMs05)])
% res06(iii,:) = [mean(PSNRs06),mean(SSIMs06)];
% disp([mean(PSNRs06),mean(SSIMs06)])
% res07(iii,:) = [mean(PSNRs07),mean(SSIMs07)];
% disp([mean(PSNRs07),mean(SSIMs07)]);
end
