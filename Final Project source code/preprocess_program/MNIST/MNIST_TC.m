%Data Preparation, by Sheng Zhang, version 1.0
%Distorted MNIST Dataset: The rotated, translated and scaled dataset(RTS)
function [TC] = MNIST_TC(Image_Matrix)
[nrow, ncol] = size(Image_Matrix);
TC=[];
for i = 1:10%ncol
    image = reshape(Image_Matrix(:,i), 28,28);
    canvas = zeros(60,60);
    
    rand_loc = randi(60-28,2);
    canvas(rand_loc(1):28+rand_loc(1)-1, rand_loc(2):28+rand_loc(2)-1) = image;
    
    six_rand = randi(ncol, 6);
    rand_loc1 = randi(28-6,6,2);
    rand_loc2 = randi(60-6,6,2);
    for j=1:6
        image = reshape(Image_Matrix(:,j),28,28);
        canvas(rand_loc2(j,1):rand_loc2(j,1)+6-1, rand_loc2(j,2):rand_loc2(j,2)+6-1)=canvas(rand_loc2(j,1):rand_loc2(j,1)+6-1, rand_loc2(j,2):rand_loc2(j,2)+6-1)+image(rand_loc1(j,1):rand_loc1(j,1)+6-1, rand_loc1(j,2):rand_loc1(j,2)+6-1);
    end
    
    TC(:,i) = reshape(canvas, 60*60,1);

end

end