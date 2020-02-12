function [K] = kernel(x1,x2,param1,param2,value)

    %When length scale is not warped,
    %param1 is matrix of size (dim,length(x1))
    %param2 is matrix of size (dim,length(x2))
    
    %eg. x1 = x, x2 = Xtrain

    x1_warped = x1.*(1./param1);
    x1_warped = x1_warped';
    x2_warped = x2.*(1./param2);
    x2_warped = x2_warped';
    nsq1=sum(x1_warped.^2,2);
    nsq2=sum(x2_warped.^2,2);
    K=bsxfun(@minus,nsq1,(2*x1_warped)*x2_warped.');
    K=bsxfun(@plus,nsq2.',K);
    K=value^2*exp(-(0.5)*K);

end