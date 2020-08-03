function s = ComputeEntropy(image,mode)
if size(image,3) == 3
    image = rgb2gray(image);
end
if mode == "entropy"
    s = entropy(double(image));
elseif mode == "entropyfilt"
    s = sum(sum(entropyfilt(image)));
end