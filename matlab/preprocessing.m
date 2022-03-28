close all;

cropSize = [32,128];
C = imread('test_er.png');

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
% ch1 - 12
% adina1 - 11
% ch2 - 15
% adina2 - 11
se = strel('disk',dil_size,0);
C_dil = imdilate(C,se);
%figure, imshow(C_dil);
% title('Dilated');

[labels, numOfWords] = bwlabel(C_dil,4);
F_word = regionprops(labels);

mkdir("swe/swe-ad");
mkdir("swe-unpadded/swe-ad");
mkdir("swe/swe-ch");
mkdir("swe-unpadded/swe-ch");
mkdir("swe/swe-so");
mkdir("swe-unpadded/swe-so");
mkdir("swe/swe-li");
mkdir("swe-unpadded/swe-li");

mkdir("samples");
mkdir("samples/samples-ad");
mkdir("samples/samples-er");
mkdir("samples/samples-ch");
for i=1:length(F_word)
    box = floor(F_word(i).BoundingBox);
    W = C_org( box(2)+dil_size:box(2)+box(4)-dil_size, box(1)+dil_size:box(1)+box(3)-dil_size );
    
    % imwrite(imbinarize(W),"swe-unpadded/swe-ad/swe-ad-ad" + sprintf('%03d',i+36) + ".png","png"); % +36 för adina, +69 för christian
    % imwrite(imbinarize(W),"swe-unpadded/swe-ch/swe-ch-ch" + sprintf('%03d',i+69) + ".png","png"); % +36 för adina, +69 för christian
    % imwrite(imbinarize(W),"swe-unpadded/swe-so/swe-so-so" + sprintf('%03d',i) + ".png","png"); % +36 för adina, +69 för christian
    % imwrite(imbinarize(W),"swe-unpadded/swe-li/swe-li-li" + sprintf('%03d',i+33) + ".png","png"); % +36 för adina, +33
    
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
    
    % imwrite(imbinarize(W),"swe/swe-ad/swe-ad-ad" + (i+36) + ".png","png"); % 
    % imwrite(imbinarize(W),"swe/swe-ch/swe-ch-ch" + (i+69) + ".png","png"); % i+69
    % imwrite(imbinarize(W),"swe/swe-so/swe-so-so" + (i+0) + ".png","png"); %
    % imwrite(imbinarize(W),"swe/swe-li/swe-li-li" + (i+33) + ".png","png"); % i+33
end