I = imread('BigData/datasets/Test/Man.jpg');
I = rgb2gray(I);
imshow(I);
E = entropyfilt(I);
E = mapminmax(E,0,1);
figure;
imshow(E);