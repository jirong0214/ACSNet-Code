dataSet = 'classic5';
folderTest  = fullfile('../../','testsets',dataSet); %%% test dataset
    %%% ��ȡͼ��·��
    ext         =  {'*.jpg','*.png','*.bmp'};
    filePaths   =  [];
    for i = 1:length(ext)   
        filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
    end
%ѡ����ģʽ��
compare_mode = 0;
if compare_mode == 1
    %ģʽ1����ԭͼ�ͳ�ʼ�ع�����������Ϣ���бȽ�
    for i = 4         %������ѡ�ļ����е�i��ͼ���saliencyֵ ���߿���ѡ���i����j�ţ�
    pic_3chanel = imread(fullfile(folderTest,filePaths(i).name)); 
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
    %pic_3chanel = imread('D:\��������\CSNet-master\CSNet-master\testsets\Set12\06.png');
    if size(pic_3chanel,3) == 3
        pic_3chanel = rgb2gray(pic_3chanel);
    end
    big_pic = double(pic_3chanel);
    block_size = 32;
    iter = 0;
    colum = im2col(big_pic,[block_size block_size],'distinct');     %ԭͼ��bxbתΪ�У��鲻�ص�������Ĳ�0��

    [a,b] = size(colum);
    S = zeros(1,b);
    
    %����������Ϣ
    for j = 1:b
        for i = 1:a
            division = 0;
            colum_xij = ones(a,1)*colum(i,j);         %����һ��ȫ�ǵ�һ����������(ij)ֵ�ľ��󣬴�СΪ���С
            d_colum = sum(abs(colum(:,j) - colum_xij));     %sum|��xkij - xij��|     %xi:�������أ�xj����Ե����
            division = division + d_colum / colum(i,j);      %sum|��xkij - xij��|/x(ij)
            Sx = 1/(block_size*block_size) * (division);        %Sxi = 1/n*sum(division)
            S(1,j) = Sx; 
        end
        iter = iter + 1
    end
end
    
    S_max = max(S);
    [m,n] = size(big_pic);
    u = m/block_size;
    v = n/block_size;
    S_im = col2im(S,[1,1],[u,v],'distinct');
    if S_max <100
        S_255 = 255*S_im/S_max;
        imshow(uint8(S_255),'InitialMagnification','fit');
        drawnow
        pause(2);
    else
        imshow((S_im),'InitialMagnification','fit');
        drawnow
        pause(2);
    end
%----------�����Ƕ�ԭͼ�ͳ�ʼ�ع�����������Ϣ���бȽ�%
else
    %ģʽ2��������һ���ļ����ڵļ���ͼƬ�������ԣ������бȽϣ�
    for i = 4         %������ѡ�ļ����е�i��ͼ���saliencyֵ ���߿���ѡ���i����j�ţ�
                      %%% ��ȡͼ�� ���� תΪ˫���ȻҶ�ͼ
        pic_3chanel = imread(fullfile(folderTest,filePaths(i).name)); 
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        %pic_3chanel = imread('D:\��������\CSNet-master\CSNet-master\testsets\Set12\06.png');
        if size(pic_3chanel,3) == 3
            pic_3chanel = rgb2gray(pic_3chanel);
        end
        big_pic = double(pic_3chanel);
        block_size = 32;
        iter = 0;
        colum = im2col(big_pic,[block_size block_size],'distinct');     %ԭͼ��bxbתΪ�У��鲻�ص�������Ĳ�0��

        [a,b] = size(colum);
        S = zeros(1,b);

        %����������Ϣ
        for j = 1:b
            for i = 1:a
                division = 0;
                colum_xij = ones(a,1)*colum(i,j);         %����һ��ȫ�ǵ�һ����������(ij)ֵ�ľ��󣬴�СΪ���С
                d_colum = sum(abs(colum(:,j) - colum_xij));     %sum|��xkij - xij��|     %xi:�������أ�xj����Ե����
                division = division + d_colum / colum(i,j);      %sum|��xkij - xij��|/x(ij)
                Sx = 1/(block_size*block_size) * (division);        %Sxi = 1/n*sum(division)
                S(1,j) = Sx; 
            end
            iter = iter + 1
        end
        S_max = max(S);
        [m,n] = size(big_pic);
        u = m/block_size;
        v = n/block_size;
        S_im = col2im(S,[1,1],[u,v],'distinct');
        if S_max <100
            S_255 = 255*S_im/S_max;
            imshow(uint8(S_255),'InitialMagnification','fit');
            drawnow
            pause(2);
        else
            imshow((S_im),'InitialMagnification','fit');
            drawnow
            pause(2);
        end
    end
end