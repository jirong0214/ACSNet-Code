slCharacterEncoding='UTF-8';
%%
% Generate a dataset of initial reconstruction
%%
clear;
t1 = clock;
global useGPU;
useGPU = 1;
ext               =  {'*.jpg','*.png','*.bmp'};
filepaths           =  [];
folder = '../BigData/datasets/BSDS500/val';
delete('SampleRate.txt');
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(folder, ext{i})));
end

for i = 1 : length(filepaths)
    image = imread(fullfile(folder,filepaths(i).name));
    if size(image,3)==3
        image = rgb2ycbcr(image);
        image = im2double(image(:, :, 1));
    else
        image =im2double(image);
    end
    image = modcrop(image,32);
    blockSize = 64;
    %Basic Sample Rate for basic sampling (0.01/0.05/0.1)
    baseInitRec = ACS_Net_initRec(image,0.05); %0.05

    %Compute a SampleRate array, and assign SR by the array;
    fun = @(block_struct)ComputeEntropy(block_struct.data); 
    Etpy  = blockproc(image,[blockSize,blockSize],fun);%Entropy matrix�?
    E1d = Etpy(:);
    E03 = roundn(E1d,-3);%3 valid numbers;
    global Strategy2EArray;
    Strategy2EArray = sort(E03);%sort the array;

    %Sampling Each block by their themselves Sample Rate;
    fun = @(block_struct) MutiScaleResidual(block_struct.data); %Basic SR 0.05;
    initEnResidual  = blockproc(image,[blockSize blockSize],fun);
    
    TotalinitRec = initEnResidual + baseInitRec; %Initial Rec = Basic initial rec + residual of additional sampling;
%     Cropedimage= size2same(TotalinitRec,image);  %resize the origin image to processed image so that can compare the ssim & psnr
     
    SIM = ssim(double(TotalinitRec),image);
    SNR = psnr(double(TotalinitRec),image);
    
    save('InitialRecSSIM.txt','SIM','-append','-ascii');
    save('InitialRecPSNR.txt','SNR','-append','-ascii');
    
     imwrite(TotalinitRec,['../BigData/datasets/DataSetForInitial2Deep/initialRecforTest/',filepaths(i).name]); 
     imwrite(image,['../BigData/datasets/DataSetForInitial2Deep/originImageforTest/',filepaths(i).name]); 
     fprintf('____________________________________________Processed images : %d\n',i);
     clear Strategy2EArray;
end
ProcessFinished = 'Process Finished ! ! ! ! !';
disp(ProcessFinished);
t2 = clock;
runtime = etime(t2,t1);