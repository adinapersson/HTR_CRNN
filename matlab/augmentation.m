% Swedish data augmentation

badFiles = ["Thumbs.db", "a01-117-05-02.png", "r06-022-03-05.png"];

num_words = 98;
num_authors = 4;
num_augmentations = 9;
createdOrNot=zeros(num_words*num_authors*num_augmentations,1);

mkdir("swe_aug");
Folders = dir('swe_segmented');
for j=3:length(Folders)
    mkdir("swe_aug/" + Folders(j).name);
    Files = dir("swe_segmented/" + Folders(j).name);
    for i=3:length(Files)
        if ~sum(contains(badFiles,Files(i).name))

            % Reading image
            C = imread(Folders(j).folder + "/" + Folders(j).name + "/" + Files(i).name);
            C = C(:,:,1);
            
            [C_stretch1, save1] = toStretch(C,1.15);
            [C_stretch2, save2] = toStretch(C,1.3);
            C_ds1 = toDilate(C_stretch1);
            C_ds2 = toDilate(C_stretch2);
            C_es1 = toErode(C_stretch1);
            C_es2 = toErode(C_stretch2);
            
            C_dilate = toDilate(C);
            C_erode = toErode(C);
            
            saveFile(C,Folders(j).name,Files(i).name,"")
            saveFile(C_dilate,Folders(j).name,Files(i).name,"di")
            saveFile(C_erode,Folders(j).name,Files(i).name,"er")
            
            if save1
                saveFile(C_stretch1,Folders(j).name,Files(i).name,"s")
                saveFile(C_ds1,Folders(j).name,Files(i).name,"ds")
                saveFile(C_es1,Folders(j).name,Files(i).name,"es")
            end
            if save2 
                saveFile(C_stretch2,Folders(j).name,Files(i).name,"ss")
                saveFile(C_ds2,Folders(j).name,Files(i).name,"dss")
                saveFile(C_es2,Folders(j).name,Files(i).name,"ess")
            end
            
            save = [1,1,1,save1,save2,save1,save2,save1,save2];
            
            for k=1:num_augmentations
                createdOrNot( (j-3)*(num_words*num_augmentations)+(i-3)*num_augmentations+k ) = save(k);
            end
            
        end
    end
end

writematrix(createdOrNot,'createdOrNot.txt')

function saveFile(C,folder,name,type)
    newName = name(1:end-4);
    newName = newName+"_"+type+".png";
    imwrite(C,fullfile("swe_aug/" + folder + "/", newName) ,'png')
end

function [C, save] = toStretch(C,stretch)
    save = 1;
    aspect_ratio = size(C,1)/size(C,2);
    if aspect_ratio > 0.25 * stretch
        C = imresize(C,[size(C,1),size(C,2)*stretch]);
    else
        save = 0;
    end
end

function C = toDilate(C)
    se = strel('disk',1,0);
    C = imdilate(C,se);
end

function C = toErode(C)
    se = strel('disk',1,0);
    C = imerode(C,se);
end
