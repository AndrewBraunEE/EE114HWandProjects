% files = dir('ece114_speech_data/*.wav');
% idx = 1;
% for file = files'
% 
%     filename = file.name;
%     [y,fs] = audioread(filename);
%     idx = idx + 1;
% end

files = dir('ece114_speech_data/*.wav');
for i = 1:170

    clear y yy speech
    [y,fs] = audioread(files(i).name);
    [yy,gg,tt,ff,zo]=v_ssubmmsev(y,fs);
    [lev,af,fso,vad]=v_activlev(yy,fs,3);
    m = 1;
    for k = 1:size(yy,1)
        if vad(k) == 1
            speech(m) = yy(k);
            m = m + 1;
        end
    end

    %y_res = resampleSINC(speech,44100/fs); 

    
    cd voiced_only
    audiowrite(strcat(files(i).name(1:end-4),'_speech.wav'),speech,fs);
    cd ..
    
end