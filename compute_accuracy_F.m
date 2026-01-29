function [accuracy,numcorrect,MacroF,MicroF,F,MCC] = compute_accuracy_F (actual,pred,classes)
% Modified by ?§×?(www.shamoxia.com) from:
% GETCM : gets confusion matrices, precision, recall, and F scores
% [confus,numcorrect,precision,recall,F] = getcm (actual,pred,[classes])
%
% actual is a N-element vector representing the actual classes
% pred is a N-element vector representing the predicted classes
% classes is a vector with the numbers of the classes (by default, it is 1:k, where k is the
%    largest integer to appear in actual or pred.
%
% dinoj@cs.uchicago.edu , Apr 2005, modified July 2005

if size(actual,1) ~= size(pred,1)
    pred=pred';
end
if nargin < 3
    classes = [1:max(max(actual),max(pred))];
end

%%
PatN=[];
MAP=0;
NDCGatN=[];
num_corr=0;
sum_cp=0;confus=[];
for i=1:length(actual)
    if actual(i)==pred(i)
        num_corr=num_corr+1;
        sum_cp=sum_cp+num_corr/i;
    end
    PatN(i)=num_corr/i;
end
if num_corr~=0
    MAP=sum_cp/num_corr;
end
%%计算MacroF,用训练集标签；
numcorrect = sum(actual==pred);
accuracy = numcorrect/length(actual);
for i=1:length(classes)
    % confus(i,:) = hist(pred,classes);
    a = classes(i);
    d = find(actual==a);     % d has indices of points with class a
    for j=1:length(classes)
        confus(i,j) = length(find(pred(d)==classes(j)));
    end
end
precision=[];
recall=[];
F=zeros(1,length(classes));
TP=[];
for i=1:length(classes)
    S=sum(confus(i,:));
    if nargout>=4
        if S
            recall(i) = confus(i,i) / sum(confus(i,:));
        else
            recall(i) = 0;
        end
    end
    S =  sum(confus(:,i));
    if nargout>=3
        if S
            precision(i) = confus(i,i) / S;
        else
            precision(i) = 0;
        end
    end
    if nargout>=5
        if (precision(i)+recall(i))
            F(i) = 2 * (precision(i)*recall(i)) / (precision(i)+recall(i));
        else
            F(i) = 0;
        end
    end
end
MacroF=mean(F);
%%
%计算MicroF，用测试集标签
%classes=unique(pred);%以测试集合为标签计算confuse矩阵
for i=1:length(classes)
    % confus(i,:) = hist(pred,classes);
    a = classes(i);
    d = find(actual==a);     % d has indices of points with class a
    for j=1:length(classes)
        confus(i,j) = length(find(pred(d)==classes(j)));
    end
end
TP=zeros(1,length(classes));FN=zeros(1,length(classes));FP=zeros(1,length(classes));TN=zeros(1,length(classes));C=sum(sum(confus));
for i=1:length(classes)
    TP(i)=confus(i,i);
    FN(i)=sum(confus(i,:))-TP(i);
    FP(i)=sum(confus(:,i))-TP(i);  
    TN(i)=C-FN(i)-TP(i)-FP(i);
    temp2(i)=(FN(i)+TP(i))*(C-FN(i)-TP(i));
    temp3(i)=(FP(i)+TP(i))*(C-FP(i)-TP(i));
end
TTP=sum(TP);TFN=sum(FN);TFP=sum(FP);
TPrecision=TTP/(TTP+TFP);TRecall=TTP/(TTP+TFN);
MicroF=2*TPrecision*TRecall/(TPrecision+TRecall);
temp1=0;
for k=1:length(classes)
    for l=1:length(classes)
        for m=1:length(classes)
            temp1=temp1+(confus(k,k)*confus(l,m)-confus(k,l)*confus(m,k));
        end
    end
end
temp2=(sum(temp2))^0.5;temp3=(sum(temp3))^0.5;
MCC=temp1/(temp2*temp3);













   


