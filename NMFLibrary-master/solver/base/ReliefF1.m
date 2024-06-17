%主函数
clear;clc;
load('data.data');
D=data(:,2:size(data,2));
m =80 ;%抽样次数
k = 8;
N=20;%运行次数
for i =1:N
    W(i,:) = RelifF (D,m,k);
end
for i = 1:N    %将每次计算的权重进行绘图,绘图N次，看整体效果
    plot(1:size(W,2),W(i,:));
    hold on ;
end
for i = 1:size(W,2)  %计算N次中，每个属性的平均值
    result(1,i) = sum(W(:,i))/size(W,1) ;
end
xlabel('属性编号');
ylabel('特征权重');
title('ReliefF算法计算乳腺癌数据的特征权重');
axis([1 10 0 0.3])
%------- 绘制每一种的属性变化趋势
xlabel('计算次数');
ylabel('特征权重');
name =char('块厚度','细胞大小均匀性','细胞形态均匀性','边缘粘附力','单上皮细胞尺寸','裸核','Bland染色质','正常核仁','核分裂');
name=cellstr(name);

for i = 1:size(W,2)
    figure
    plot(1:size(W,1),W(:,i));
    xlabel('计算次数') ;
    ylabel('特征权重') ;
    title([char(name(i))  '(属性' num2Str(i) ')的特征权重变化']);
end