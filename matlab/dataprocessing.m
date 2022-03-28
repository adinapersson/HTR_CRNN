% Create folder for the processed data
cropSize = [32,128];
 
tic
mkdir("full_data");
badFiles = ["Thumbs.db", "a01-117-05-02.png", "r06-022-03-05.png"];
 
Folders = dir('words');
for j=3:length(Folders)
    mkdir("full_data\" + Folders(j).name);
    subFolders = dir("words\" + Folders(j).name);
    for k=3:length(subFolders)
        mkdir("full_data\" + Folders(j).name + "\" + subFolders(k).name)
        Files = dir("words\" + Folders(j).name + "\" + subFolders(k).name);
        for i=3:length(Files)
            if ~sum(contains(badFiles,Files(i).name))
                frame = uint8(zeros(cropSize));
 
                % Reading image
                C = imread(Folders(j).folder + "\" + Folders(j).name + "\" + subFolders(k).name + "\" + Files(i).name);
                
                % Inverting
                C = 255-C;
                
                % Filtering
                C = medfilt2(C,[3,3]);
                
                % Thresholding
                thresh = ceil(255*graythresh(C));
                C(C<thresh) = 0;
                
                % Intensity normalisation
                peak = im2double(max(max(C)));
                C = C/peak;
                C = uint8(C);
                
                % Scaling and padding
                scale = size(frame,1)/size(C,1);
                C = imresize(C,scale);
                frame(:,1:size(C,2)) = C;
                C = frame;
                C = imresize(C,cropSize);      
 
                % Write to file
                imwrite(C,fullfile("full_data\" + Folders(j).name + "\" + subFolders(k).name + "\", Files(i).name),'png')
            end
        end
    end
end
toc