%Data Preparation, by Sheng Zhang, version 1.0
%Distorted MNIST Dataset: The rotated, translated and scaled dataset(RTS)
function [RTS] = MNIST_R(Image_Matrix)
[nrow, ncol] = size(Image_Matrix);
RTS=[];
for i = 1:ncol
    image = reshape(Image_Matrix(:,i), 28,28);
    angle = -45 + (45-(-45))*rand(1,1);
    image = imrotate(image,angle,'bilinear','crop');
    
    scale = 0.7 + (1.2-0.7)*rand(1,1);
    image = imresize(image,scale);
    sz = size(image,1);
    
    canvas = zeros(42,42);
    rand_loc = randi(42-sz,2);
    canvas(rand_loc(1):sz+rand_loc(1)-1, rand_loc(2):sz+rand_loc(2)-1) = image;
    RTS(:,i) = reshape(canvas, 42*42,1);

end

end