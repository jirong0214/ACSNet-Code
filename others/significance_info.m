dataSet = 'classic5';
folderTest  = fullfile('../../','testsets',dataSet); %%% test dataset
    %%% 读取图像路径
    ext         =  {'*.jpg','*.png','*.bmp'};
    filePaths   =  [];
    for i = 1:length(ext)   
        filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
    end
%选择工作模式：
compare_mode = 0;
if compare_mode == 1
    %模式1：对原图和初始重构的显著性信息进行比较
    for i = 4         %计算所选文件夹中第i张图像的saliency值 或者可以选择第i到第j张；
    pic_3chanel = imread(fullfile(folderTest,filePaths(i).name)); 
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
    %pic_3chanel = imread('D:\下载内容\CSNet-master\CSNet-master\testsets\Set12\06.png');
    if size(pic_3chanel,3) == 3
        pic_3chanel = rgb2gray(pic_3chanel);
    end
    big_pic = double(pic_3chanel);
    block_size = 32;
    iter = 0;
    colum = im2col(big_pic,[block_size block_size],'distinct');     %原图按bxb转为列，块不重叠，不足的补0；

    [a,b] = size(colum);
    S = zeros(1,b);
    
    %求显著性信息
    for j = 1:b
        for i = 1:a
            division = 0;
            colum_xij = ones(a,1)*colum(i,j);         %构造一个全是第一个中心像素(ij)值的矩阵，大小为块大小
            d_colum = sum(abs(colum(:,j) - colum_xij));     %sum|（xkij - xij）|     %xi:中心像素；xj：边缘像素
            division = division + d_colum / colum(i,j);      %sum|（xkij - xij）|/x(ij)
            Sx = 1/(block_size*block_size) * (division);        %Sxi = 1/n*sum(division)
            S(1,j) = Sx; 
        end
        iter = iter + 1
    end
end
    
    S_max = max(S);
    [m,n] = size(big_pic);
    u = m/block_size;
    v = n/block_size;
    S_im = col2im(S,[1,1],[u,v],'distinct');
    if S_max <100
        S_255 = 255*S_im/S_max;
        imshow(uint8(S_255),'InitialMagnification','fit');
        drawnow
        pause(2);
    else
        imshow((S_im),'InitialMagnification','fit');
        drawnow
        pause(2);
    end
%----------以上是对原图和初始重构的显著性信息进行比较%
else
    %模式2：仅计算一个文件夹内的几幅图片的显著性，不进行比较；
    for i = 4         %计算所选文件夹中第i张图像的saliency值 或者可以选择第i到第j张；
                      %%% 读取图像 并且 转为双精度灰度图
        pic_3chanel = imread(fullfile(folderTest,filePaths(i).name)); 
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        %pic_3chanel = imread('D:\下载内容\CSNet-master\CSNet-master\testsets\Set12\06.png');
        if size(pic_3chanel,3) == 3
            pic_3chanel = rgb2gray(pic_3chanel);
        end
        big_pic = double(pic_3chanel);
        block_size = 32;
        iter = 0;
        colum = im2col(big_pic,[block_size block_size],'distinct');     %原图按bxb转为列，块不重叠，不足的补0；

        [a,b] = size(colum);
        S = zeros(1,b);

        %求显著性信息
        for j = 1:b
            for i = 1:a
                division = 0;
                colum_xij = ones(a,1)*colum(i,j);         %构造一个全是第一个中心像素(ij)值的矩阵，大小为块大小
                d_colum = sum(abs(colum(:,j) - colum_xij));     %sum|（xkij - xij）|     %xi:中心像素；xj：边缘像素
                division = division + d_colum / colum(i,j);      %sum|（xkij - xij）|/x(ij)
                Sx = 1/(block_size*block_size) * (division);        %Sxi = 1/n*sum(division)
                S(1,j) = Sx; 
            end
            iter = iter + 1
        end
        S_max = max(S);
        [m,n] = size(big_pic);
        u = m/block_size;
        v = n/block_size;
        S_im = col2im(S,[1,1],[u,v],'distinct');
        if S_max <100
            S_255 = 255*S_im/S_max;
            imshow(uint8(S_255),'InitialMagnification','fit');
            drawnow
            pause(2);
        else
            imshow((S_im),'InitialMagnification','fit');
            drawnow
            pause(2);
        end
    end
end