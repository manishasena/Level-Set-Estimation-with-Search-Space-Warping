function [mu_s,std2] = build_GP(Xtrain,ytrain,X_test,param,param_training,value,noise)

    %% Updating all Covariance Matricies
    %Covariance K_ss
    K_ss = kernel(X_test,X_test,param,param,value);

    %Kernal for training data 
    K = kernel(Xtrain,Xtrain,param_training,param_training,value);

    %K_star
    K_s = kernel(Xtrain,X_test,param_training,param,value);

    %Updated Mean + noise
    mu_s = K_s'/(K+ noise^2*eye(length(K)))*ytrain;

    %Updated Covariance and Sigma + noise
    sigma_s = K_ss - K_s'/(K + noise*eye(length(K)))*K_s;
    std2 = real(sqrt(diag(sigma_s)));

end