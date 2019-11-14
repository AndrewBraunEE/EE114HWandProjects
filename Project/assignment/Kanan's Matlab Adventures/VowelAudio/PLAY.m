files = dir('*.wav');
for file = files'
    [sound,f] = audioread(file.name);
    soundsc(sound,f);
    pause(.08);
    plot(sound)
    hold on
end