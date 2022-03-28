% Segmentation and preprocessing (withpout rescaling and padding) of the Swedish data (swe_segmented is used in augmentation.m)
close all;

mkdir("swe_segmented");
filenames = ["ad1.jpg", "ad2.jpg", "ch1.jpg", "ch2.jpg", "li1.png", "li2.png", "so.jpg"];
save_names = ["swe_segmented/swe-ad/swe-ad-ad", "swe_segmented/swe-ad/swe-ad-ad", "swe_segmented/swe-ch/swe-ch-ch",...
    "swe_segmented/swe-ch/swe-ch-ch", "swe_segmented/swe-li/swe-li-li", "swe_segmented/swe-li/swe-li-li",...
    "swe_segmented/swe-so/swe-so-so"];
dil_size = [11 11 12 15 11 11 11];
offsets = [0 36 0 69 0 33 0];
cropSize = [32,128];

for j=1:length(filenames)
    C = imread("swe_images/" + filenames(j));
    C_org = C;
    
    % Inverting
    C = 255-C;
    C = rgb2gray(C); % grey scale
    
    % Filtering
    C = medfilt2(C,[3,3]);

    % Thresholding
    thresh = ceil(255*graythresh(C));
    C(C<thresh) = 0;
    C(C>=thresh) = 255;
    
    se = strel('disk',dil_size(j),0);
    C = imdilate(C,se);

    [labels, numOfWords] = bwlabel(C,4);
    F_word = regionprops(labels);

    mkdir("swe_segmented/swe-ad");
    mkdir("swe_segmented/swe-ch");
    mkdir("swe_segmented/swe-so");
    mkdir("swe_segmented/swe-li");
    for i=1:length(F_word)
        box = floor(F_word(i).BoundingBox);
        W = C_org( box(2)+dil_size(j):box(2)+box(4)-dil_size(j), box(1)+dil_size(j):box(1)+box(3)-dil_size(j) );
    
        imwrite(W,save_names(j) + sprintf('%03d',i+offsets(j)) + ".png","png");
    end
    
end 
