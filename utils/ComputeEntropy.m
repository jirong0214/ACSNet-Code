function s = ComputeEntropy(image)
%image = imread('BigData/datasets/Test/Fish.jfif');
if size(image(3)) == 3
    image = rgb2gray(image);
end
E = entropyfilt(image);
%E = mapminmax(E,0,1);
s = sum(sum(E));
% blockSize = 32;
%  fun = @(block_struct) entropyfilt(block_struct.data); 
%  Etpy  = blockproc(fullimage,[blockSize,blockSize],fun);
% % E1d = Etpy(:);%获得了一个N行1列的熵向量；
% E03 = roundn(E1d,-3);
% E = sort(E03);
% %index = find(E1d == E);
end