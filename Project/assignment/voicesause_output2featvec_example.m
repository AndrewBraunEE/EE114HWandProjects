files = dir('*.mat');

feat_vec=[];
labels=[];
for file = files'

    load(file.name);
    disp(file.name)
    use_data=[mean(remove_nan(sF0)) mean(remove_nan(sF1)) mean(remove_nan(sF2))...
        mean(remove_nan(sF3)) mean(remove_nan(sF4))...
        mean(remove_nan(H2KH5Kc)) mean(remove_nan(Energy))...
        mean(remove_nan(A1)) mean(remove_nan(A2)) mean(remove_nan(A3))...
        mean(remove_nan(CPP)) mean(remove_nan(F2K))];
    
    feat_vec=[feat_vec; use_data];
    labels=[labels; contains(file.name,'english')];
    clearvars -except feat_vec files file labels
end


function output = remove_nan(input)
output=input(~isnan(input));
end
