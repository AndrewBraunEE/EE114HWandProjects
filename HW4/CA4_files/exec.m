%soundsc(tone,8000);
Fs = 32000;

load_in();
figure();
specgram(tone);
xlabel('Time');
ylabel('Frequency (Hz)');
title('Spectrogram of 8000 Hz Tone');

%soundsc(noise,8000);
figure();
specgram(noise);
xlabel('Time');
ylabel('Frequency (Hz)');
title('Spectrogram of Noise');

%soundsc(sweep,8000);
figure();
specgram(sweep);
xlabel('Time');
ylabel('Frequency (Hz)');
title('Spectrogram of Sweep');


NFFT = 512

L = 480;
Nol = 360;

soundsc(male_sentence,8000);
figure();
%Fs = 192000;
specgram(male_sentence, NFFT, Fs, L, Nol);
xlabel('Frequency');
ylabel('Frequency (Hz)');
title('Spectrogram of Male Sentence (480, 360)');

L = 240;
Nol = 180;

soundsc(male_sentence,8000);
figure();
specgram(male_sentence, NFFT, Fs, L, Nol);
xlabel('Frequency');
ylabel('Frequency (Hz)');
title('Spectrogram of Male Sentence (240, 180)');

L = 60;
Nol = 45;

soundsc(tone,8000);
figure();
specgram(male_sentence, NFFT, Fs, L, Nol);
xlabel('Frequency');
ylabel('Frequency (Hz)');
title('Spectrogram of Male Sentence (60, 45)');

%Linear Prediction Order

figure();
load_in;

zpfft(male_a, 8000, 1);
title('zpfft male_a 8000');
xlabel('Frequency');
ylabel('Amplitude');

p = 4;
x = male_a;
a = lpc(x, p);

%H_z = Gain / a;
num = 1;
den = a;

figure;
subplot(2,1,1);
[h, w] = freqz(1, a);
plot(w, 20*log10(abs(h)));
xlabel('Frequency (Radians)');
ylabel('Amplitude');
title('Spectral Representation LP p = 4');

subplot(2,1,2);
zplane(num, den);
xlabel('Real');
ylabel('Imag');
title('Pole-Zero plot  LPC p = 4');

p = 6;
a = lpc(x, p);

num = 1;
den = a;

%H_z = Gain / a;

figure;
subplot(2,1,1);
[h, w] = freqz(1, a);
plot(w, 20*log10(abs(h)));
xlabel('Frequency (Radians)');
ylabel('Amplitude');
title('Spectral Representation LP p = 6');

subplot(2,1,2);
zplane(num, den);
xlabel('Real');
ylabel('Imag');
title('Pole-Zero plot  LPC p = 6');

p = 8;
a = lpc(x, p);
%H_z = Gain / a;

num = 1;
den = a;

figure;
subplot(2,1,1);
[h, w] = freqz(1, a);
plot(w, 20*log10(abs(h)));
xlabel('Frequency (Radians)');
ylabel('Amplitude');
title('Spectral Representation LP p = 8');

subplot(2,1,2);
zplane(num, den);
xlabel('Real');
ylabel('Imag');
title('Pole-Zero plot  LPC p = 8');

p = 10;
a = lpc(x, p);

%H_z = Gain / a;

num = 1;
den = a;

figure;
subplot(2,1,1);
[h, w] = freqz(1, a);
plot(w, 20*log10(abs(h)));
xlabel('Frequency (Radians)');
ylabel('Amplitude');
title('Spectral Representation LP p = 10');

subplot(2,1,2);
zplane(num, den);
xlabel('Real');
ylabel('Imag');
title('Pole-Zero plot  LPC p = 10');

p = 12;
a = lpc(x, p);

%H_z = Gain / a;

num = 1;
den = a;

figure;
subplot(2,1,1);
[h, w] = freqz(1, a);
plot(w, 20*log10(abs(h)));
xlabel('Frequency (Radians)');
ylabel('Amplitude');
title('Spectral Representation LP p = 12');

subplot(2,1,2);
zplane(num, den);
xlabel('Real');
ylabel('Imag');
title('Pole-Zero plot  LPC p = 12');

p = 100;
a = lpc(x, p);

num = 1;
den = a;

%H_z = Gain / a;

figure;
subplot(2,1,1);
[h, w] = freqz(1, a);
plot(w, 20*log10(abs(h)));
xlabel('Frequency (Radians)');
ylabel('Amplitude');
title('Spectral Representation LP p = 100');

subplot(2,1,2);
zplane(num, den);
xlabel('Real');
ylabel('Imag');
title('Pole-Zero plot  LPC p = 100');

