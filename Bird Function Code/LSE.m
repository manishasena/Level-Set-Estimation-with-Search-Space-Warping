function [x_next] = LSE(Xtrain,ytrain,x_sample,h,param,param_training,value,noise,iteration,lb,ub)
  
    x0 = x_sample;

    fun = @(x)acq_optimisation_LSE(Xtrain,ytrain,x,h,param,param_training,value,noise,iteration);

    rng default 
    gs = GlobalSearch('NumTrialPoints',100,'NumStageOnePoints',100,'Display','off');
    options = optimoptions('fmincon');
    problem = createOptimProblem('fmincon','x0',x0,...
        'objective',fun,'lb',lb,'ub',ub,'options',options);
    x = run(gs,problem);
    x_next = x;

end