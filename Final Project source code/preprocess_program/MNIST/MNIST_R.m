%Data Preparation, by Sheng Zhang, version 1.0
%Distorted MNIST Dataset: The rotated dataset(R)
function [R] = MNIST_R(Image_Matrix)
[nrow, ncol] = size(Image_Matrix);
R=[];
for i = 1:ncol
    image = reshape(Image_Matrix(:,i), 28,28);
    angle = -90 + (90-(-90))*rand(1,1);
    image = imrotate(image,angle,'bilinear','crop');
    image = imresize(image, [42,42]);
    R(:,i) = reshape(image, 42*42,1);

end

end