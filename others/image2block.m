file_path =  'C:\Users\Melbourne\OneDrive - stu.ouc.edu.cn\代码\ACSNet\ACSNet\ACSNet-Code\Set5\';% 图像文件夹路径
img_path_list = dir(strcat(file_path,'*.bmp'));%获取该文件夹中所有jpg格式的图像
img_num = length(img_path_list);%获取图像总数量
if img_num > 0 %有满足条件的图像
        for k = 1:img_num %逐一读取图像
            image_name = img_path_list(k).name;% 图像名
            Im=  imread(strcat(file_path,image_name));
            L = size(Im);
            height=64;
            width=64;
            max_row = floor(L(1)/height);
            max_col = floor(L(2)/width);
            seg = cell(max_row,max_col);
            %分块
            for row = 1:max_row      
                for col = 1:max_col        
                seg(row,col)= {Im((row-1)*height+1:row*height,(col-1)*width+1:col*width,:)};  
                end
            end 
 
            
            for i=1:max_row*max_col
                save_dir_final = ['C:\Users\Melbourne\OneDrive - stu.ouc.edu.cn\代码\ACSNet\ACSNet\ACSNet-Code\blockimage\',image_name,'\']
                if exist(fullfile(save_dir_final),'dir')==0
                    mkdir(fullfile(save_dir_final));
                end  
            imwrite(seg{i},strcat(save_dir_final,'m',int2str(i),'_',image_name));  
            end
%            % 画出分块的边界
%             for row = 1:max_row      
%                 for col = 1:max_col  
%              rectangle('Position',[160*(col-1),160*(row-1),160,160],...
%                      'LineWidth',2,'LineStyle','-','EdgeColor','r');
%                     end
%             end 
%             hold off


        end
end