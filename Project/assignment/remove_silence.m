% files = dir('ece114_speech_data/*.wav');
% idx = 1;
% for file = files'
% 
%     filename = file.name;
%     [y,fs] = audioread(filename);
%     idx = idx + 1;
% end

files = dir('ece114_speech_data/*.wav');
for i = 55:55

    clear y yy speech
    [y,fs] = audioread(files(i).name);
    %[yy,gg,tt,ff,zo]=v_ssubmmsev(y,fs);
    [lev,af,fso,vad]=v_activlev(y,fs,3);
    m = 1;
    for k = 1:size(y,1)
        if vad(k) == 1
            speech(m) = y(k);
            m = m + 1;
        end
    end

    %y_res = resampleSINC(speech,44100/fs); 

    
    cd voiced_only
    audiowrite(strcat(files(i).name(1:end-4),'_speech.wav'),speech,fs);
    cd ..
    
end