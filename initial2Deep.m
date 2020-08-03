function DeepRecImage = initial2Deep(InitialRecImage)
    netfolder = '../BigData/model/Init2DeepModel/BSDS500_Initial2Deep_bSize64_patch96_stride32';% tianjirong's trained model folder；
    netpaths = dir(fullfile(netfolder,'net-epoch-100.mat')); % switch net path
    net = load(fullfile(netfolder,netpaths.name)); 
    net = dagnn.DagNN.loadobj(net.net);
    global useGPU;
    %InitialRecImage =im2single(InitialRecImage);
    
    if useGPU
        net.move('gpu');
        input = gpuArray(InitialRecImage);
    else
        input = InitialRecImage;
    end

    net.conserveMemory = false;
    net.eval({'input',input});

    if useGPU
            DeepRecImage = gather(net.vars(net.getVarIndex('dr_prediction')).value);
    else
            DeepRecImage = net.vars(net.getVarIndex('dr_prediction')).value;    %(deep reconstruction prediction);
    end
end