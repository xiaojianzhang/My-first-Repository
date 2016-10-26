%Data Preparation, by Sheng Zhang, version 1.0
%Distorted MNIST Dataset: The projected dataset(P)
function [P] = MNIST_P(Image_Matrix)
[nrow, ncol] = size(Image_Matrix);
P=[];
for i = 1:%ncol
    image = reshape(Image_Matrix(:,i), 28,28);
    scale = 0.75 + (1.0-0.75)*rand(1,1);
    image = imresize(image, scale);
    sz = size(image,1);
    sample = abs(normrnd(0,5));
    stretch = round(sample*sqrt(2)*0.5);
    image = imresize(image, [2*stretch+sz, 2*stretch+sz]);
    image = imresize(image, [42,42]);
    P(:,i) = reshape(image, 42*42,1);

end

end