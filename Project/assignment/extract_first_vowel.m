filename = 'armenian3.wav';
[y,fs] = audioread(filename);
[yy,gg,tt,ff,zo]=v_ssubmmsev(y,fs);

windowed_y = yy(1:100000);
s = specgram(windowed_y);
threshold = 10;
% specgram(windowed_y);
% figure();
mesh(abs(s));
flag = 0;
start = 0;
k = 1;
while k <= 780
   if flag == 0
      if abs(s(1,k)) >= threshold || abs(s(2,k)) >= threshold || abs(s(3,k)) >= threshold || abs(s(4,k)) >= threshold
          start = k;
          flag = 1;
      end
   end
   if flag == 0 && start == 0 && k == 780
       threshold = threshold - 1;
       k = 1;
   end
   k = k + 1;
end

start_sample = round(start/405 * 52000);
vowel = y(start_sample:start_sample + 2048);
soundsc(vowel,fs);

cd FirstVowels
save(strcat(filename(1:end-4),'Vowe1'), 'vowel')
cd ..
