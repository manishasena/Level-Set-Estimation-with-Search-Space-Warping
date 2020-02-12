function [LSE_warped_val] = acq_optimisation_LSEwarped(Xtrain,ytrain,x_sample,h,param,param_training,norm_length,U1_length,value,noise,iteration)
    %tic
    %Build function GP
    [mu_s,std2] = build_GP(Xtrain,ytrain,x_sample,param,param_training,value,noise);
       
    %% Warping the length scale based on distance of mean to h 
    B = sqrt(1*2*log(iteration^(1+2)*pi^2/(3*0.01))); 
    
    param_warp = log(1+ (abs(mu_s - h)+0.1).^2./((B*std2+0.1).^2)); %was 0.01
    param_warp = norm_length.*param_warp' + U1_length.*ones(size(x_sample));

    param_training_warp = log(1+ (abs(ytrain-h)+0.1).^2/((0.1^2))); 
    param_training_warp = norm_length.*param_training_warp' + U1_length.*ones(size(Xtrain));
    
    %Build warped GP for LSE to sample from
    [mu_s,std2] = build_warpedGP(Xtrain,ytrain,x_sample,param_warp,param_training_warp,value,noise);     
    %disp("LSE_warped : " + toc)
    %% Choosing next point
    
    %LSE_warped values for all points
    LSE_warped_val = -1*abs(mu_s - h) + B*std2;
    LSE_warped_val = -1*LSE_warped_val;

end