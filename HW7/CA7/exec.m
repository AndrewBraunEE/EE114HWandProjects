im = imread('ca6_image.tiff');
Ug = imnoise(im,'gaussian',0,0.002);
Us = imnoise(im,'salt & pepper');

h_H = (1/7).*[-1 -2 -1; -2 19 -2; -1 -2 -1]

%High Pass

Uh = imfilter(im,h_H);

%Histogram Equalization

Ue = histeq(im);

%LPF
h = (1/10).*[1 1 1; 1 2 1; 1 1 1]
Uf = imfilter(Ug, h)

%Median
Um = medfilt2(Us);

figure();
imshow(Uh);
title('Highpass')

figure();
imshow(Ue);
title('Histogram')

figure();
imshow(Uf);
title('LPF')

figure();
imshow(Um);
title('Median')

figure();
imshow(im);
title('Original')

figure();
imshow(Ug);
title('Gaussian')

figure();
imshow(Us);
title('SaltPepper')