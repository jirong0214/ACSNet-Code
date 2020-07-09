addpath('BigData');
clear;
t1 = clock;
%OriginImage = imread('BigData/datasets/Test/Fish.jfif');
OriginImage = imread('BigData/datasets/Test/Man.jpg');
%OriginImage = imread('/Users/tianjirong/OneDrive - stu.ouc.edu.cn/����/ACSNet/ACSNet/ACSNet-Code/RecResult/����1/Baby/baby_GT.bmp');
%OriginImage = imread('/Users/tianjirong/OneDrive - stu.ouc.edu.cn/����/ACSNet/ACSNet/ACSNet-Code/RecResult/����1/butterfly/butterfly_GT.bmp');
if size(OriginImage,3)==3
    OriginImage = rgb2ycbcr(OriginImage);
    OriginImage = im2double(OriginImage(:, :, 1));  %����ͼ��תΪ�Ҷȣ�
else
    OriginImage =im2double(OriginImage);
end
OriginImEntropy = entropy(OriginImage);
delete('SampleRate.txt');
PSNR = 0;
blockSize =64;
%switch work mode;
fun = @(block_struct)ComputeEntropy(block_struct.data); 
Etpy  = blockproc(OriginImage,[blockSize,blockSize],fun);%�ؾ���
%Etpy  = Etpy1(1:size(Etpy1,1)-1,1:size(Etpy1,2)-1);
E1d = Etpy(:);%�����һ��N��1�е���������
E03 = roundn(E1d,-3);%����3λС��
global EArray;
EArray = sort(E03);%��������
global mode;
mode = 2; % 1Ϊ�ֿ鴦�������ʸ���ͼ���ط��䣻2Ϊ����ͼ��ȫ��һ�鴦��3.�ԱȲ��ԣ�
global ConstRate;%���ù̶��Ĳ����ʣ�
ConstRate = 0.2;
switch mode 
    case 1  %��ΪbSize*bSize�Ŀ鴦��
        fun = @(block_struct) ACS_Net_finalOutput(block_struct.data); 
        result  = blockproc(OriginImage,[blockSize blockSize],fun);
        %imwrite(result,'./Test/results/CurryBlocked_block64_SR_range0.01-0.5.bmp');
        filename = 'SampleRate.txt';
        SRs=textread(filename,'%f');
        AvgSampleRate = mean(SRs);
        imshow(result);
        result = double(result);
        OriginImage = size2same(result,OriginImage);  %��ԭͼ�ü�Ϊ�ͽ��ͼһ����ģ�
        PSNR = psnr(result,OriginImage); 
        SSIM = ssim(result,OriginImage); 
        fprintf('��ͬ����䲻ͬ������ʱ���ؽ���PSNR =%g\n',PSNR);
        fprintf('��ͬ����䲻ͬ������ʱ���ؽ���SSIM  =%g\n',SSIM);
        fprintf('ƽ�������� = %g\n',AvgSampleRate);
        %imwrite(result,'/Users/tianjirong/OneDrive - stu.ouc.edu.cn/����/ACSNet/ACSNet/ACSNet-Code/RecResult/Fish/����1/Fish_avgSR0093.jpg'); 

    case 2   %����ͼ����
        result= ACS_Net_finalOutput(OriginImage);  
        imshow(result);
        result = double(result);
        OriginImage = size2same(result,OriginImage);  %��ԭͼ�ü�Ϊ�ͽ��ͼһ����ģ�
        SSIM = ssim(result,OriginImage); 
        PSNR = psnr(result,OriginImage); 
        fprintf('����ͼ��ֱ�Ӱ�ͬһ�������ʲ���ʱ���ؽ���PSNR =%g\n',PSNR);
        fprintf('����ͼ��ֱ�Ӱ�ͬһ�������ʲ���ʱ���ؽ���SSIM =%g\n',SSIM);
        
    case 3    %����ͼ���� +��Ϊ64*64�Ŀ鴦��
        result_full = ACS_Net_Function(OriginImage); 
        fun = @(block_struct) ACS_Net_Function(block_struct.data);  
        result_blocked = blockproc(OriginImage,[64,64],fun);
        
        result_full = double(result_full);
        result_blocked = double(result_blocked);
        
        OriginImage = size2same(result,OriginImage);  %��ԭͼ�ü�Ϊ�ͽ��ͼһ����ģ�
        PSNR_full = psnr(result_full,OriginImage); 
        PSNR_blocked = psnr(result_blocked,OriginImage); 
        
        imshow(cat(2,im2uint8(OriginImage),im2uint8(result_full),im2uint8(result_blocked)));
        fprintf('%g',PSNR_full,'%g',PSNR_blocked);
end

t2 = clock;
runtime = etime(t2,t1);
E = ComputeEntropy(OriginImage);
fprintf("���п��ƽ���ش�С��%d \n",mean(EArray(:)));
fprintf("������Ϊ��%d",EArray);
