%Nwn = 200;
one;
Nwn = 40;
Q1 = time_analysis(data,'rectwin',Nwn,1);
Q2 = time_analysis(data,'rectwin',Nwn,2);
Q3 = time_analysis(data,'rectwin',Nwn,3);
Qa = time_analysis(a,'rectwin',Nwn,3); 
Qsh = time_analysis(sh,'rectwin',Nwn,3);

%Autocorrelation
Ra = xcorr(a,a);
Rsh = xcorr(sh,sh);
AMDFa = amdf(a);
AMDFsh = amdf(sh);


figure;
subplot(4,1,1);
plot(Ra)
xlabel('sample number')
ylabel('Amplitude')
title('Autocorrelation of /a/');
subplot(4,1,2);
plot(Rsh)
xlabel('sample number')
ylabel('Amplitude')
title('Autocorrelation of /sh/');
subplot(4,1,3);
plot(AMDFa);
xlabel('sample number');
ylabel('Amplitude');
title('AMDF /a/');
subplot(4,1,4);
plot(AMDFsh);
xlabel('Sample Number');
ylabel('Amplitude');
title('AMDF /sh/');

%Part 2
figure;
load_in;
zpfft(male_a,8000,8);
title('Male A');
figure;
zpfft(female_a,8000,8);
title('Female A');
%Change Nw here
Nw = 90;

figure;
zpfft(female_a(1:Nw),8000,8);
title('Windowed Female A');
figure;
Nw = 50;
zpfft(male_a(1:Nw),8000,8);
title('Windowed Male A');

