
%image = imread('BigData/datasets/Test/Fish.jfif');
image = imread('BigData/datasets/Test/Man.jpg');
image = rgb2gray(image);
blockSize = 64;
fun = @(block_struct) ComputeEntropy(block_struct.data); 
Etpy1 = blockproc(image,[blockSize,blockSize],fun)
Etpy  = Etpy1(1:size(Etpy1,1)-1,1:size(Etpy1,2)-1);
% % E1d = Etpy(:);%获得了一个N行1列的熵向量；
% E03 = roundn(E1d,-3);
% E = sort(E03);
% %index = find(E1d == E);
%Etpy = mapminmax(Etpy,0,255);
imshow(image);
figure;
imshow(Etpy);