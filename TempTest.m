image = imread('/Users/tianjirong/OneDrive - stu.ouc.edu.cn/代码/ACSNet/BigData/datasets/Set5/baby_GT.bmp');
I = image(65:128,65:128);
E = ComputeEntropy(I);