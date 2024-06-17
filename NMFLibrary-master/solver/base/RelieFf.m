%Relief函数实现
%D为输入的训练集合,输入集合去掉身份信息项目;k为最近邻样本个数
function W = ReliefF (D,m,k);
Rows = size(D,1) ;%样本个数
Cols = size(D,2) ;%特征熟练,不包括分类列
　　type2 = sum((D(:,Cols)==2))/Rows ;
　　type4 = sum((D(:,Cols)==4))/Rows ;
　　%先将数据集分为2类，可以加快计算速度
　　D1 = zeros(0,Cols) ;%第一类
　　D2 = zeros(0,Cols) ;%第二类
　　for i = 1:Rows
　　    if D(i,Cols)==2
　　        D1(size(D1,1)+1,:) = D(i,:) ;
　　    elseif D(i,Cols)==4
　　        D2(size(D2,1)+1,:) = D(i,:) ;
　　    end
　　end
　　W =zeros(1,Cols-1) ;%初始化特征权重，置0
　　for i = 1 : m  %进行m次循环选择操作
　　   %从D中随机选择一个样本R
　　    [R,Dh,Dm] = GetRandSamples(D,D1,D2,k) ;
　　    %更新特征权重值
　　    for j = 1:length(W) %每个特征累计一次，循环
　　        W(1,j)=W(1,j)-sum(Dh(:,j))/(k*m)+sum(Dm(:,j))/(k*m) ;%按照公式更新权重
　　    end
　　end