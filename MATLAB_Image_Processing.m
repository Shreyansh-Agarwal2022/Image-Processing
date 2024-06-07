% DIP Project Code
% Name : Shreyansh Agarwal
% Roll no. : 22ucs203

%   TASK 1 : Convert RGB image to Gray Scale image

% importing image 'img3.jpg'
RGB_image = imread('img3.jpg');

% split rgb channel (using array slicing)
R = RGB_image(:,:,1);
G = RGB_image(:,:,2);
B = RGB_image(:,:,3);

% Making Gray Level Matrix from R,G,B matrices, weights are self-defined
gray = 0.298 * R + 0.590 * G + 0.110 * B;

imshow(gray);
save_gray_image(gray,"output1.jpg");

% Free memory for matrix/array not used further
clear R;
clear G;
clear B;
clear RGB_image;

%   TASK 2 : Add 30% salt and pepper noise on it. 
%   (15% Salt noise, 15% Pepper Noise)

total_pixels = numel(gray);
noise_pixels = round(0.15 * total_pixels);  % 15% of total pixels  

% randperm() function generates a vector of random permutation of integers
% Generate random indices for salt noise
salt_indices = randperm(total_pixels, noise_pixels);

% Generate random indices for pepper noise
pepper_indices = randperm(total_pixels, noise_pixels);

% Adding salt noise
for i = 1:length(salt_indices)
    gray(salt_indices(i)) = 255;
end

% Adding pepper noise
for i = 1:length(pepper_indices)
    gray(pepper_indices(i)) = 0;
end

imshow(gray);
save_gray_image(gray,"output2.jpg");

% Free memory for matrix/array not used further
clear salt_indices;
clear pepper_indices;

%   TASK 3 : Remove noise using adaptive median filter

% extracting no. of rows and columns in the image
[rows, cols] = size(gray);

% making a new matrix for saving the restored images
% zeros() funtion is used to fill the matrix with 0
result_image = zeros(rows, cols, 'uint8');
for i = 1:rows
    for j = 1:cols
        % Calling adaptive median fitler function to get new gray level
        % value for each pixel
        result_image(i,j) = adaptive_median_filter(gray,i,j,3,rows,cols);
    end
end

imshow(result_image);
gray = result_image;
save_gray_image(gray,"output3.jpg");

% Free memory for matrix/array not used further
clear result_image;

%   TASK 4 : Use Otsu's thresholding technique for making the thresholding image of it.

% Getting the threshold value for our gray scale image
T = otsu_thresholding(gray);
%disp(T);

% converting image into binary of foreground and background
result_image = zeros(rows,cols);
for i = 1:rows
    for j = 1:cols
        if gray(i,j) > T
            result_image(i,j) = 1;
        end
    end
end

gray = result_image;
% Free memory for matrix/array not used further
clear result_image;

imshow(gray);
save_gray_image(gray,"output4.jpg");

%   TASK 5 : Use Morphological Analysis to remove small objects.

% First erosion to remove small objects in background and then dilate to
% remove small objects in foreground

result_image = zeros(rows,cols);
for i = 1:rows
    for j = 1:cols
        result_image(i,j) = erosion(gray,i,j,rows,cols,9);
    end
end
gray = result_image;
clear result_image;

result_image = zeros(rows,cols);
for i = 1:rows
    for j = 1:cols
        result_image(i,j) = dilation(gray,i,j,rows,cols,13);
    end
end

gray = result_image;
% Free memory for matrix/array not used further
clear result_image;
imshow(gray);
save_gray_image(gray,"output5.jpg");

%   TASK 6 : Use Morphological Analysis to find the largest connected component.

% Connect all the foreground objects using dilation and then perform
% erosion to get the foreground in correct area

result_image = zeros(rows,cols);
for i = 1:rows
    for j = 1:cols
        result_image(i,j) = dilation(gray,i,j,rows,cols,11);
    end
end
gray = result_image;
% Free memory for matrix/array not used further
clear result_image;

result_image = zeros(rows,cols);
for i = 1:rows
    for j = 1:cols
        result_image(i,j) = erosion(gray,i,j,rows,cols,9);
    end
end
gray = result_image;
% Free memory for matrix/array not used further
clear result_image;
imshow(gray);
save_gray_image(gray,"output6.jpg");

%   TASK 7 : Find the area of it in form of pixels.

% To get area of foreground in pixels count the number of foreground pixels
% in binary image
result = 0;
for i = 1:rows
    for j = 1:cols
        if gray(i,j) == 1
            result = result + 1;
        end
    end
end

disp('Area (in pixels) = ');
disp(result);

%   Adding Roll no. 203 to image at the end resulting image

% Add the text
textPosition = [10, 10]; % Define the position where you want to add the text (top-left corner)
textString = '203'; % The text you want to add
result_image = insertText(gray, textPosition, textString, 'FontSize', 12, 'TextColor', 'white', 'BoxOpacity', 0);

% Display the image with the text
imshow(result_image);
gray = result_image;
% Free memory for matrix/array not used further
clear result_image;
save_gray_image(gray,"result_image.jpg");


function save_gray_image(gray, file_name)
    % Get the current working directory
    current_folder = pwd;

    file_path = fullfile(current_folder, file_name);

    % Write the gray matrix as an image
    temp=gray;
    imwrite(temp, file_path);
    % Free memory for matrix/array not used further
    clear temp;
end

%{

% Padding funcion for adding zero padding in image but not used in the image

function padded_image = zeroPadding(gray, pad)
    image=gray;
    [rows, cols] = size(image);
    
    % Create a new matrix for the padded image
    padded_image = zeros(rows+2*pad, cols+2*pad, 'uint8');
    
    % Copy the original image into the padded image
    for i = 1:rows
        for j = 1:cols
            padded_image(i+pad, j+pad) = image(i, j);
        end
    end
end
%}

% Returns the new gray level value for the pixel(x,y) of the noisy image

function res = adaptive_median_filter(gray,x,y,windowSize,rows,cols)
    S_max = 11;          % '9' is good for not-so-high-end-CPU

    % window for perform for getting the median, max, min for the
    % pixel(x,y) from the image
    window=ones(1, windowSize*windowSize, 'uint8');
    half=floor(windowSize/2);
    temp=1;
    value = int16(gray(x,y));

    % Filling the window
    for i = x-half:x+half
        for j = y-half:y+half
            if i<=0 || i>rows || j<=0 || j>cols
                window(temp) = 0;
            else
                window(temp) = gray(i,j);
            end
            temp = temp + 1;
        end
    end

    % Sorting the window to get min, max, median
    window = sort(window);
    z_min = int16(window(1));
    z_max = int16(window(end));
    z_median = int16(window(floor(numel(window)/2)+1));

    % Level A
    A1 = z_median - z_min;
    A2 = z_median - z_max;
    B1 = value - z_min;
    B2 = value - z_max;
  
    %disp([windowSize,x,y,A1,A2,z_min,z_max,z_median]);
    %disp(window);
    %disp([x,y,A1,A2,B1,B2,z_min,z_max,z_median]);
    if A1>0 && A2<0
        % Level B
        if B1>0 && B2<0
            res = gray(x,y);
        else
            res = z_median;
        end
    else
        % Level A
        windowSize = windowSize + 2;
        if windowSize<=S_max
            res = adaptive_median_filter(gray,x,y,windowSize,rows,cols);
        else
            res = gray(x,y);
        end
    end
    % Free memory for matrix/array not used further
    clear window;
end


% Otsu Threshold function gives value of threshold to divide image in
% foreground and background
function threshold = otsu_thresholding(gray)
    [rows, cols] = size(gray);

    % Initialize an array to store the count of each gray level
    gray_level_counts = zeros(1, 256, 'single');

    % Iterate through the image and count the occurrences of each gray level
    for i = 1:rows
        for j = 1:cols
            % Add 1 because MATLAB indexing starts from 1
            gray_level_counts(gray(i, j) + 1) = gray_level_counts(gray(i, j) + 1) + 1;
        end
    end
    
    % Calculate the probability of each gray level
    total_pixels = single(numel(gray));
    p = gray_level_counts / total_pixels;
    % Free memory for matrix/array not used further
    clear gray_level_counts;

    max = 0;        % Initialize maximum to zero
    sigma = zeros(1, 255, 'single');
    P1 = sum(p(1:1));       % Probability of class 1
    P2 = sum(p(2:256));     % Probability of class 2

    for T = 2:255
        P1 = P1 + p(T);
        P2 = P2 - p(T);
        
        mu1 = sum((0:T-1) .*p(1:T))/P1;     % class mean u1
        mu2 = sum((T:255) .*p(T+1:256))/P2;     % class mean u2
        %disp([mu1,mu2])
        sigma(T) = P1*P2*((mu1-mu2)^2);     %compute simga, i.e., variance (between class)
        if sigma(T)>max     % compare sigma with maximum
            max = sigma(T);     % update max value
            threshold = T-1;        % desired threshold correspond to maximum value of between class
        end
    end
end

% Erosion function which gives a new value for pixel(x,y) to your binary image
function result = erosion(binary,x,y,rows,cols,windowSize)
    half=floor(windowSize/2);
    temp = 0;       % temp count the pixels in foreground in the structural element at pixel(x,y)
    for i = x-half:x+half
        for j = y-half:y+half
            if i<=0 || i>rows || j<=0 || j>cols
                break;      % the structrual element is not coming fully in the image
            else
                if binary(i,j)==1
                    temp = temp + 1;        % increment temp because foreground pixel found
                else
                    break;                  % break loop because background pixel found
                end
            end
        end
    end
    if temp == windowSize*windowSize
        result = 1;                 % new value of pixel(x,y) is foreground
    else
        result = 0;                 % new value of pixel(x,y) is background
    end
end

% Dilation function which gives a new value for pixel(x,y) to your binary image
function result = dilation(binary,x,y,rows,cols,windowSize)
    half=floor(windowSize/2);
    result = 0;                     % by default result of pixel(x,y) is background
    for i = x-half:x+half
        for j = y-half:y+half
            if i<=0 || i>rows || j<=0 || j>cols
                continue;
            else
                if binary(i,j)==1     % if any pixel in window is from foreground
                    result = 1;     % new value of pixel(x,y) is foreground
                else
                    continue;
                end
            end
        end
    end
end
