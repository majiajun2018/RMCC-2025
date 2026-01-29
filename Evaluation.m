function [accuracy,MacroF,MicroF,F,MCC]=Evaluation(W,A,test_fea,test_label)
Class_number=size(W,2);
actual=test_label;Test_number=length(actual);
x=[test_fea,ones(Test_number,1)];
predict=zeros(Test_number,1);
for i=1:Test_number
    temp=zeros(Class_number,1);
    P=repmat(W'*x(i,:)',[1,Class_number])-eye(Class_number);
    for j=1:Class_number  
        temp(j)=P(:,j)'*A*P(:,j);
    end
%     f=x(i,:)*W;
%  [~,predict(i)]=max(f);   
 [~,predict(i)]=min(temp);
end
[accuracy,~,MacroF,MicroF,F,MCC]  = compute_accuracy_F (actual,predict,(1:Class_number));
end