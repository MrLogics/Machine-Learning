X=iris;
MDL=fitcnb(X,Y,'ClassNames',{'Iris-setosa', 'Iris-versicolor', 'Iris-verginia'}, 'CrossVal', 'on', 'kfold', 5);
[label,score]=predict(MDL.Trained{4}, XTest);
YPred=categorical(label);
confusionmat(YTest,YPred);
accuracy=sum(YPred==YTest)/numel(YTest)