function [W,A,train_time,Y,Z] = RMCCtrain(train_fea,train_label,lamda,a)
nu=1e-6;
X=[train_fea ones(size(train_fea,1),1)];[Train_number,d]=size(X);
Class_number=length(unique(train_label));Y=zeros(Train_number,Class_number);
for j=1:Class_number
    Y(train_label==j,j)=1;
end
tic
W=(X'*X+lamda*eye(d))\X'*Y;
G=zeros(Class_number,Class_number);S=zeros(Class_number,Class_number);
Z=X*W;H=Z-Y;
for j=1:Class_number
    C=H(train_label==j,:);n_j=size(C,1);
    D=H(train_label~=j,:);
    G=G+C'*C;
    S=S+D'*D;
end
G=G+nu*eye(Class_number);S=S+nu*eye(Class_number);
A= CS_GeometricMean(inv(G),S,a);
 %da=(diag(A)).^0.5;A=A./repmat(da,1,Class_number);A=A./repmat(da',Class_number,1);
%A=A./norm(A,'fro');%A=A./repmat(diag(A),1,Class_number);
% A=(G^(-1/2))*(((G^(1/2))*S*(G^(1/2)))^a)*(G^(-1/2));
%A=(G/S);
train_time=toc;
end
