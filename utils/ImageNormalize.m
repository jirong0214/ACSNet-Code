function result = ImageNormalize(image)
    %This function convert an image to a specific size that can be mod by 32;
    if size(image(3)) == 3
        image = rgb2gray(image);
    end
    Size= size(image);
    row = Size(1);
    col = Size(2);
    rowCut = mod(row,32);
    colCut = mod(col,32);
    result = image(1:(row-rowCut),1:(col-colCut));
end
