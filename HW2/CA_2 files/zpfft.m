function []=zpfft(data,fs,zpfactor)
len=length(data);
fft_length=len*zpfactor;
half_length=fft_length/2+1;
fa=20*log10(abs(fft(data,fft_length)));
subplot(2,1,1);
plot(data);
title('Time Domain zpFFT');
xlabel('Samples');
ylabel('Amplitude');
subplot(2,1,2);
plot(linspace(0,fs/2,half_length),fa(1:half_length));
title('Freq Domain zpFFT');
xlabel('Frequency (Hz)');
ylabel('dB Mag');
%axis([1,half_length,0.9*min(fa),1.1*max(fa)]);
