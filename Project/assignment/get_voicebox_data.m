addpath('VoiceBox\voicebox');
mfcc_vec = [];
mfcc_cell = {};
fbe_vec = [];
fbe_cell = {};
labels = [];
files = dir('ece114_speech_data\*.wav');
Fs = 44100;
for k = 1:length(files)
    file = "ece114_speech_data\" + files(k).name;
    data = v_readwav(convertStringsToChars(file));
    mfcc = v_melcepst(data, Fs, '0dD');
    fb = v_melbankm(floor(3 * log(Fs)), pow2(floor(log2(0.03 * Fs))), Fs);
    fbe = log(sum(fb.^2, 2));
    mfcc_cell{k} = mfcc;
    fbe_cell{k} = fbe;
    labels = [labels; contains(files(k).name, 'english')];
    fprintf("Processed %d/%d\n", k, length(files));
end

shortest = min(cellfun('size', mfcc_cell, 1));
for i = 1:length(mfcc_cell)
    mfcc_vec(:, :, i) = resample(mfcc_cell{i}, shortest, size(mfcc_cell{i}, 1));
end
save('mfcc_feat_vec.mat', 'mfcc_vec');
shortest = min(cellfun('size', fbe_cell, 1));
for i = 1:length(fbe_cell)
    fbe_vec(:, :, i) = resample(fbe_cell{i}, shortest, size(fbe_cell{i}, 1));
end
save('fbe_feat_vec.mat', 'fbe_vec');
save('labels.mat', 'labels');

disp('Done');