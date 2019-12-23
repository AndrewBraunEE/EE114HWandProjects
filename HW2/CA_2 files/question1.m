load_in;
a = 0.8
y = [1 -a];
freqz(y, 1);
out=filter([1 -0.8],1,female_sentence);
%Compare audio of pre-emphasis filtering
soundsc(female_sentence,8000);
soundsc(out,8000);
%Compare Spectrogram:
figure(1);
subplot(2,1,1);
spectrogram(female_sentence,100,[],[],8000,'yaxis');
subplot(2,1,2);
spectrogram(out,100,[],[],8000,'yaxis');
legend('Unfiltered', 'Filtered');
title('Filtered Female vs Unfiltered');
xlabel('Frequency');
ylabel('dB');
hold off;
saveas(gcf,'Spectrogram.png')
%Display Logmagnitude and filtered steady state /a
figure(2);
zpfft(female_a,8000,10);
hold on;
out_a=filter([1 -0.8],1,female_a);
zpfft(out_a,8000,10);
title('Filtered Female vs Unfiltered Steady State');
xlabel('Frequency');
ylabel('dB');
legend('Unfiltered', 'Filtered');
hold off;
saveas(gcf,'Filter_Female_SteadyState.png')

y = [[1, -0.8], [1, -0.5], [1, 0.8]];
y0 = [1, -0.8];
y1 = [1, -0.5];
y2 = [1, 0.8];
figure(3);
zplane(y0, 1);
title('a=0.8');
xlabel('Real');
ylabel('Imag');
saveas(gcf,'a_08.png')
figure(4);
zplane(y1, 1);
title('a=0.5');
xlabel('Real');
ylabel('Imag');
saveas(gcf,'a_05.png')
figure(5);
zplane(y2, 1);
title('a=-0.8');
xlabel('Real');
ylabel('Imag');
saveas(gcf,'a-0.8.png')

%Question 2
%Determine the approximate spectral locations for the first three for-
%mant frequencies.
figure(6);
zpfft(male_a,8000,10);
title('Formant Frequencies for Male');
xlabel('Frequency');
ylabel('dB');
legend('Male Voice');
saveas(gcf,'FormantMale.png')

figure(7);
%Question 3, Pitch Period Estimation
subplot(2, 1, 1);
plot(male_a);axis tight;
title('Male Pitch Period Est');
xlabel('Frequency');
ylabel('dB');
subplot(2,1,2);
plot(female_a);axis tight;
title('Female Pitch Period Est');
saveas(gcf,'MaleFemalePitchPeriod.png')
xlabel('Frequency');
ylabel('dB');