% Segmentation and preprocessing of classification samples
close all;

cropSize = [32,128];
C = imread('classification_samples/test_er.png');

% Inverting
C = 255-C;
C = rgb2gray(C); % grey scale

% Filtering
C = medfilt2(C,[3,3]);

C_org = C;

% Intensity normalisation
peak = im2double(max(max(C_org)));
C_org = C_org/peak;
C_org = uint8(C_org);

% Thresholding
thresh = ceil(255*graythresh(C));
C(C<thresh) = 0;
C(C>=thresh) = 255;

dil_size = 12;
se = strel('disk',dil_size,0);
C_dil = imdilate(C,se);

[labels, numOfWords] = bwlabel(C_dil,4);
F_word = regionprops(labels);

mkdir("samples");
mkdir("samples/samples-ad");
mkdir("samples/samples-er");
mkdir("samples/samples-ch");
for i=1:length(F_word)
    box = floor(F_word(i).BoundingBox);
    W = C_org( box(2)+dil_size:box(2)+box(4)-dil_size, box(1)+dil_size:box(1)+box(3)-dil_size );
    
    % Scaling and padding
    frame = uint8(zeros(cropSize));
    scale = size(frame,1)/size(W,1);
    W = imresize(W,scale);
    frame(:,1:size(W,2)) = W;
    W = frame;
    W = imresize(W,cropSize); 
    
    W = 255-W;
    
    %imwrite(W,"samples/samples-ad/samples-ad-" + i + ".png","png");
    imwrite(W,"samples/samples-er/samples-er-" + i + ".png","png");
    %imwrite(W,"samples/samples-ch/samples-ch-" + i + ".png","png");
end
