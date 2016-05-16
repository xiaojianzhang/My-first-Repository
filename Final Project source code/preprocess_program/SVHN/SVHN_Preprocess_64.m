%SVHN Dataset Preprocessing, by Sheng Zhang, version 1.0
%64 * 64 crops around each digit sequece
function [Image, label] = SVHN_Preprocess_64(digitStruct)
Image = [];
label = 10*ones(length(digitStruct),5);
for i = 1:length(digitStruct)
    im = imread([digitStruct(i).name]);
    [height, width, channel] = size(im);
    
    for j = 1:length(digitStruct(i).bbox)
        aa(j) = max(digitStruct(i).bbox(j).top+1,1);
        bb(j) = min(digitStruct(i).bbox(j).top+digitStruct(i).bbox(j).height, height);
        cc(j) = max(digitStruct(i).bbox(j).left+1,1);
        dd(j) = min(digitStruct(i).bbox(j).left+digitStruct(i).bbox(j).width, width);
        label(i,j) = digitStruct(i).bbox(j).label;
    end
    aa = min(aa);
    bb = max(bb);
    cc = min(cc);
    dd = max(dd);
    
    x_expand = 0.15*(bb-aa);
    y_expand = 0.15*(dd-cc);
    aa = max(1,aa-x_expand);
    bb = min(height, bb+x_expand);
    cc = max(1,cc-y_expand);
    dd = max(width,dd+y_expand);
    im = im(aa:bb, cc:dd,:);
    im = imresize(im,[64,64]);
    Image(:,i) = reshape(im,64*64*3,1);
        
end
end
