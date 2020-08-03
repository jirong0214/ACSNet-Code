global useGPU;
global showDetailMsg;
useGPU = 1;
showDetailMsg = 0;
saveImage = 1;
Size = [];
for imageNo = 829:890
    image = imread(['../BigData/datasets/reference-890/GroundTruth/','image',num2str(imageNo),'.png']);
    Size = size(image);
    if Size(1)>1920 || Size(2) >1080
        useGPU = 0;
    else
        useGPU = 1;
    end
    [psnr_totalInit_cur,ssim_totalInit_cur,psnr_deep_cur,ssim_deep_cur,AvgSampleRate_cur] = ACSNetEval(image,imageNo,saveImage);%1 means save rec result;
    %Save Quality matrix
    save('../BigData/datasets/reference-890/psnr_totalInit.txt','psnr_totalInit_cur','-append','-ascii');
    save('../BigData/datasets/reference-890/ssim_totalInit.txt','ssim_totalInit_cur','-append','-ascii');
    save('../BigData/datasets/reference-890/psnr_deep.txt','psnr_deep_cur','-append','-ascii');
    save('../BigData/datasets/reference-890/ssim_deep.txt','ssim_deep_cur','-append','-ascii');
    save('../BigData/datasets/reference-890/AvgSampleRate.txt','AvgSampleRate_cur','-append','-ascii');
    disp(['_______________Processed',num2str(imageNo),'images!_______________']);
end



    %Compute avgs by saved txts;
    psnrs_totalInit = '../BigData/datasets/reference-890/psnr_totalInit.txt';
    avg_psnr_totalInit=mean(textread(psnrs_totalInit,'%f'));
    
    ssims_totalInit = '../BigData/datasets/reference-890/ssim_totalInit.txt';
    avg_ssim_totalInit=mean(textread(ssims_totalInit,'%f'));
    
    psnrs_deep = '../BigData/datasets/reference-890/psnr_deep.txt';
    avg_psnr_deep=mean(textread(psnrs_deep,'%f'));
    
    ssims_deep = '../BigData/datasets/reference-890/ssim_deep.txt';
    avg_ssim_deep=mean(textread(ssims_deep,'%f'));
    
    SampleRates = '../BigData/datasets/reference-890/AvgSampleRate.txt';
    avgSampleRate =mean(textread(SampleRates,'%f'));
    
    %Save in Summary txt by below order
    save('../BigData/datasets/reference-890/Summary.txt','avg_psnr_totalInit','-append','-ascii');
    save('../BigData/datasets/reference-890/Summary.txt','avg_ssim_totalInit','-append','-ascii');
    save('../BigData/datasets/reference-890/Summary.txt','avg_psnr_deep','-append','-ascii');
    save('../BigData/datasets/reference-890/Summary.txt','avg_ssim_deep','-append','-ascii');
    save('../BigData/datasets/reference-890/Summary.txt','avgSampleRate','-append','-ascii');


