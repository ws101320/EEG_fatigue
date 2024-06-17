clc;
clearvars;
close all;
dataFolder = 'D:\python\pythonProject3\BCI_learn\11111\NMF\rank1_data\';
dataFolder1 = 'D:\python\pythonProject3\SEED_VIG\EEG_Feature_2Hz\';
files = dir([dataFolder '*.mat']);
files1 = dir([dataFolder1 '*.mat']);
fileNumbers = arrayfun(@(x) sscanf(x.name, '%d'), files);
fileNumbers1 = arrayfun(@(x) sscanf(x.name, '%d'), files1);
[~, sortedIndices] = sort(fileNumbers);
[~, sortedIndices1] = sort(fileNumbers1);
sortedFiles = files(sortedIndices);
sortedFiles1 = files1(sortedIndices1);
for s = 1:23
    load([dataFolder sortedFiles(s).name]);
    load([dataFolder1 sortedFiles1(s).name]);
    fprintf('Processing file: %s\n', sortedFiles(s).name);
    fprintf('Processing file: %s\n', sortedFiles1(s).name);
    De = data;
    De = reshape(De, [17, 25, 885]);
    Value = zeros(17, 25, 885);
    value = reshape(value,[885,17])
    for i = 1:885
        for j = 1:17
            Value(j, :, i) = value(i, j) .* De(j, :, i);
        end
    end
    Value = reshape(Value, [17, 885, 25]);
    save(['D:\python\pythonProject3\BCI_learn\11111\NMF\5_bands\' num2str(s) '.mat'], 'Value');
end
