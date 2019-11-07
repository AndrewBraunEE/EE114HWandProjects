files = dir('FirstVowels/*.mat');
idx = 1;
for file = files'
    data = load(file.name);
    cd VowelAudio
        audiowrite(strcat(file.name(1:end-4),'_audio.wav'),data.vowel,fss(idx));
    cd ..
    idx = idx + 1;
end
