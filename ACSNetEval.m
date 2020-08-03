function [psnr_totalInit,ssim_totalInit,psnr_deep,ssim_deep,AvgSampleRate] = ACSNetEval(image, imageNo,saveResult)
    %this global param controls the block image order;
    delete('SampleRate.txt');
    global blockNo; 
    blockNo = 0;
    %blocksize of blockproc;
    blockSize = 64;
    %%
    if size(image,3)==3
        image = rgb2ycbcr(image);
        image = im2double(image(:, :, 1));
    else
        image = im2double(image);
    end
    % convert image -> can be mod by 32;
    image =modcrop(image,32); 

    %%
    %base Sampling by SR of 0.05;
    %基础采样部分：
    baseInitRec = ACS_Net_initRec(image,0.05); 

    %%
    %compute Entropy matrix by baseInitRec！！
    %根据初始重构分块计算图像熵：
    %注意：使用"entropy"时,使用entropy函数来计算熵; 使用"entropyfilt"时,使用entropyfilt函数来计算熵; 两者结果可能不同；
    fun = @(block_struct)ComputeEntropy(block_struct.data,"entropy");
    entropyMatrix = blockproc(baseInitRec,[blockSize,blockSize],fun);
    %convert the entropyMatrix to a 1D entropyVector;
    entropyVector = getEntropyVector(entropyMatrix);
    %sort the entropyVector;
    sortedEntropyVector = sort(entropyVector);

    %%
    %补充采样部分：
    %分块分不同采样率补充采样,获得残差；这里需要用到entropyVector和sortedEntropyVector;
    fun = @(block_struct)MutiScaleResidual(block_struct.data,entropyVector,sortedEntropyVector); 
    initEnResidual  = blockproc(image,[blockSize blockSize],fun);
    %基准初始重构+补充采样的残差=恢复的初始重构
    TotalinitRec = initEnResidual + baseInitRec; 

    %%
    %deep Reconstruction!
    %深度重构：
    deepReconstruction = initial2Deep(TotalinitRec);

    %%
    %evaluate the quality of the result  && show result
        %[psnr_base,ssim_base]             = Cal_PSNRSSIM(im2uint8(image) ,im2uint8(baseInitRec),0,0);
        [psnr_totalInit,ssim_totalInit]  = Cal_PSNRSSIM(im2uint8(image) ,im2uint8(TotalinitRec),0,0);
        [psnr_deep,ssim_deep]            = Cal_PSNRSSIM(im2uint8(image) ,im2uint8(deepReconstruction),0,0);


        %display the average Sample Rate;
        filename = 'SampleRate.txt';
        SRs=textread(filename,'%f');
        AvgSampleRate = mean(SRs);
    %%
    %save Initial Result && Deep Result!
    if saveResult == 1
        imwrite(TotalinitRec,              ['../BigData/datasets/reference-890/InitialResult/','image',num2str(imageNo),'.png']); 
        imwrite(deepReconstruction,['../BigData/datasets/reference-890/DeepResult/','image',num2str(imageNo),'.png']); 
    end
end
