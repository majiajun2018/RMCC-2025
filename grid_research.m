%% MSG
bestcv=0;
for lamda=10.^(-3:1:4)
    for a=10.^(-6:1:-1)
        [W,A,train_time,Y] = RMCCtrain(train_fea,train_label,lamda,a);
        [accuracy,MacroF,MicroF,F,MCC]=Evaluation(W,A,test_fea,test_label);
        if (accuracy>bestcv)
           bestlamda=lamda;besta=a;bestcv=accuracy;
        end
   end
end