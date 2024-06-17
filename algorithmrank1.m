clc; 
clearvars; 
close all; 
options.alg = 'mu';
rank = 1;
dataFolder='D:\python\pythonProject3\SEED_VIG\EEG_Feature_2Hz\';
files=dir([dataFolder '*.mat']);
fileNumbers = arrayfun(@(x) sscanf(x.name, '%d'), files);
[~, sortedIndices] = sort(fileNumbers);
sortedFiles = files(sortedIndices);
for s = 1:length(sortedFiles)
    filePath = [dataFolder sortedFiles(s).name];
    load(filePath);
    fprintf('Processing file: %s\n', sortedFiles(s).name);
    De = de_LDS;
    value = zeros(17,885);
for num = 1:885
    de = De(:,num,:);
    de = squeeze(de);
    V = cov(de');
    [x, infos] = ns_nmf(V, rank, options);
    Diff = (x.W)*(x.H)-V;
    A = (x.H);
    vmin = min(A);
    vmax = max(A);
for i = 1:rank
   A(i,:) = (A(i,:)-vmin(i))/(vmax(i)-vmin(i));
   value(:,num) = A(i,:);
end
end
save(['D:\python\pythonProject3\BCI_learn\11111\rank1_weight\\' num2str(s) '.mat'],'value');
end
