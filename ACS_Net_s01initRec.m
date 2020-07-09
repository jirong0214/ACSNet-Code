function s01Init = ACS_Net_s01initRec(image) %���s01��ĳ�ʼ�ع���
tic
%image = imread('./datasets/Test/Submarine.jfif');

%%ѡ��ѵ����ģ��·����
%netfolder = '.\model'; %pretrained model��
%netpaths = dir(fullfile(netfolder,['net',ratio,'.mat']));%pretrained��  
netfolder = 'testmodel';% tianjirong's trained model folder��
netpaths = dir(fullfile(netfolder,['net-epoch-20.mat']));  % switch net path
net = load(fullfile(netfolder,netpaths.name)); 
net = dagnn.DagNN.loadobj(net.net);

%ͼ��תΪ�����ȻҶ�ͼ��
if size(image,3)==3
    image = rgb2ycbcr(image);
    image = im2single(image(:, :, 1));
else
    image =im2single(image);
end
image = modcrop(image,32); 
input = image;
net.conserveMemory = false;
net.eval({'input',input});

s01Init          = net.vars(net.getVarIndex('s01combine')).value;

toc
end