% This function convert EntropyMatrix to a EntropyVector; and the vector satisfies the sampling order;
function res = getEntropyVector(EntropyMatrix)

    EntropyMatrixSize = size(EntropyMatrix);
    row = EntropyMatrixSize(1);
    col  = EntropyMatrixSize(2);
    res = [];
    %1st: process four corner;
    res = [res,EntropyMatrix(1,1)];
    res = [res,EntropyMatrix(1,col)];
    res = [res,EntropyMatrix(row,1)];
    res = [res,EntropyMatrix(row,col)];
    %2nd: process the first row (excepts two corner)
    res = [res,EntropyMatrix(1,2:col - 1)];
    for i = 2:row-1
        res = [res,EntropyMatrix(i,1:col - 1)];
    end
    %3rd:process the last row (excepts two corner)
    res = [res,EntropyMatrix(row,2:col - 1)];
    %4rd:process the right col elements(excepts two corner)
    for j = 2:row-1
        res = [res,EntropyMatrix(j,col)];
    end
    
end


    