addpath('VOICEBOX');
addpath('ece114_speech_data/ece114_speech_data/');
feat_vec_rnn=[];
feat_cell={};
labels=[];
files = dir('ece114_speech_data/ece114_speech_data/*.wav')
count=1;
for file = files'
    disp(file.name);
    [audio,fs]=audioread(file.name);
    mfccs=v_melcepst(audio,fs);
    feat_cell{count}=mfccs;
count=count+1;
labels=[labels; contains(file.name,'english')];
end
%%
shortest=min(cellfun('size',feat_cell,1));

for ii = 1:length(feat_cell)
    feat_vec_rnn(:,:,ii)=resample(feat_cell{ii},shortest,size(feat_cell{ii},1));
end