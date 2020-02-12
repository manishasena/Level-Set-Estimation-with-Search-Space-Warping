function [LSE] = acq_optimisation_LSE(Xtrain,ytrain,x_sample,h,param,param_training,value,noise,iteration)
    
    %tic
    %Build function GP
    [mu_s,std2] = build_GP(Xtrain,ytrain,x_sample,param,param_training,value,noise);
    %disp("LSE: " + toc)
    B = sqrt(1*2*log(iteration^(1+2)*pi^2/(3*0.01))); 
    
    %LSE values for all points
    LSE = -1*abs(mu_s - h) + B*std2;
    LSE = -1*LSE;

end