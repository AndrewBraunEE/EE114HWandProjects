files = dir('ece114_speech_data/*.wav');
idx = 1;
for file = files'

filename = file.name;
[y,fs] = audioread(filename);
fss(idx,1) = fs;
idx = idx + 1;
% [yy,gg,tt,ff,zo]=v_ssubmmsev(y,fs);
% 
% windowed_y = yy(1:200000);
% s = specgram(windowed_y);
% threshold = 5;
% specgram(windowed_y);
% figure();
% mesh(abs(s));
% flag = 0;
% start = 0;
% k = 1;
% while k <= 1561
%    if flag == 0
%       if abs(s(1,k)) >= threshold || abs(s(2,k)) >= threshold || abs(s(3,k)) >= threshold || abs(s(4,k)) >= threshold
%           start = k;
%           flag = 1;
%       end
%    end
%    if flag == 0 && start == 0 && k == 1561
%        threshold = threshold - 1;
%        k = 1;
%    end
%    k = k + 1;
% end
% 
% start_sample = round(start/1561 * 200000);
% vowel = y(start_sample+1000:start_sample + 3048);
% soundsc(vowel,fs);
% 
% cd FirstVowels
% save(strcat(filename(1:end-4),'Vowe1'), 'vowel')
% cd ..
 
end
