function A =  size2same(B,A)
[ra ca] = size(A);
[rb cb] = size(B);
if ra > rb
    ra = rb;
end
if ca > cb
    ca = cb;
end
A = imresize(A,[ra ca]);


