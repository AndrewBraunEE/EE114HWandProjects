addpath('VoiceBox\voicebox');
mfcc_vec = [];
mfcc_cell = {};
fbe_vec = [];
fbe_cell = {};
labels = [];
files = dir('ece114_speech_data\*.wav');
for k = 1:length(files)
    file = "ece114_speech_data\" + files(k).name;
    [data, Fs] = v_readwav(convertStringsToChars(file));
    data = v_ssubmmsev(data, Fs);
    mfcc = v_melcepst(data, Fs, 'dDE');
    for i = 1:size(mfcc, 2)
       mfcc(:, i) = mfcc(:, i) - mean(mfcc(:, i)); 
    end
    mfcc_cell{k} = mfcc;
    labels = [labels; contains(files(k).name, 'english')];
    fprintf("Processed %d/%d\n", k, length(files));
end

shortest = min(cellfun('size', mfcc_cell, 1));
for i = 1:length(mfcc_cell)
    mfcc_vec(:, :, i) = resample(mfcc_cell{i}, shortest, size(mfcc_cell{i}, 1));
end
save('mfcc_feat_vec.mat', 'mfcc_vec');
save('labels.mat', 'labels');

disp('Done');