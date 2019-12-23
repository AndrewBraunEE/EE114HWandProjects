function hw
	function FAmp = ScaleFTAmp(F)
		%Shift FT with shift, take log scale

		c = 6.;
		temp = log10(abs(F)/max(max(abs(F))) + 0.000001);
		%Normalize to 255
        %disp(FAmp);
		temp = (temp + c)*255./c;
		FAmp = fftshift(temp);
    end

	function out = ReduceFT(in, range)
		%Reduce FT coefficients to near-dc components
		[x_size, y_size] = size(in);
		out = in;
		out(range + 1 : x_size - range, :) = 0.0001;
		out( :, range+1 : x_size - range) = 0.0001;
    end

	im = imread('ca6_image.tiff');
	%a_fft = fft2(im)
	%im = ifft2(a_fft)

	%Part 1: Display the image and its FT
    figure();
	imagesc(im);
	colormap(gray);
	axis square;

	%Display FT
    figure();
	ft = fft2(im);
	sft = ScaleFTAmp(ft);
	imagesc(sft);
	colormap(gray);
	axis square;

	%Part 2: Reduce to -128:127, FT Coefficients Magnitude
	figure();
    ft128 = ReduceFT(ft, 128);
	sft128 = ScaleFTAmp(ft128);
	imagesc(sft128);
	colormap(gray);
	axis square;

	%Part 3: Reconstructingbridge from -128, 127
    figure();
	im128 = real(ifft2(ft128));
	imagesc(im128);
	axis square;
	psnr(im, im128)

	%Repeat this for -64:63 and -32:31
	figure();
    ft64 = ReduceFT(ft, 64);
	sft64 = ScaleFTAmp(ft64);
	imagesc(sft64); colormap(gray);
	axis square;

	%Reconstruct image
	im64 = real(ifft2(ft64));
	imagesc(im64);
	axis square;
	psnr(im, im64)

	%32 case
	figure();
    ft32 = ReduceFT(ft, 32);
	sft32 = ScaleFTAmp(ft32);
	imagesc(sft32); colormap(gray);
	axis square;

	%reconstruct brige
	figure();
    im32 = real(ifft2(ft32));
	imagesc(im32);
	axis square;
	psnr(im, im32)
end

%hw()