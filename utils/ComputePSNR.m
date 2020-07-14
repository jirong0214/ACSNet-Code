Oimage = imread('/Users/tianjirong/OneDrive - stu.ouc.edu.cn/代码/ACSNet/ACSNet/BigData/datasets/Test/Fish.jfif') ;
Rimage1 = imread('/Users/tianjirong/OneDrive - stu.ouc.edu.cn/代码/ACSNet/ACSNet/ACSNet-Code/RecResult/Fish/策略1/Fish_avgSR0089-PSNR18.10-SSIM0.6608.jpg');
Rimage2 = imread('/Users/tianjirong/OneDrive - stu.ouc.edu.cn/代码/ACSNet/ACSNet/ACSNet-Code/RecResult/Fish/按相同采样率整体重构/Fish_full_0.1_PSNR18.25_SSIM0.668_.jpg');
Oimage = rgb2gray(Oimage);
Oimage = size2same(Rimage2,Oimage); 
PSNR = psnr(Rimage2,Oimage)