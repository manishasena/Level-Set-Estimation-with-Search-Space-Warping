
function [H_length_warped] = bird_warping(U1_length1, U1_length2, norm_length1, norm_length2, h)

    
    x_points = 41;
    y_points = 41;
    x_lowerlimit = -4;
    y_lowerlimit = -4;
    x_upperlimit = 4;
    y_upperlimit = 4;
    
    %Lower and Upper limits
    lb = [x_lowerlimit,y_lowerlimit];
    ub = [x_upperlimit,y_upperlimit];

    %True length parameters
    default_length = [1.391;1.5478]; 
    
    %Initial Sample points
    n_sample = 3;
   
    %Number of test points to classify
    n_sample_test = 50;
    
    %Intialiser
    n_sample_SAMPLING = 1;
    
    %Kernel parameter
    value = 1;
    noise = 0;
    
    no_trials = 3;
    no_iterations = 20;
    
    %Store accuracy
    H_length_warped = 0;
    
    for e = 1:no_trials
    
        Xtrain_original = zeros(2,n_sample);
        Xtrain_original(1,:) = (x_upperlimit - x_lowerlimit).*rand(1,n_sample) + x_lowerlimit;
        Xtrain_original(2,:) = (y_upperlimit - y_lowerlimit).*rand(1,n_sample) + y_lowerlimit;
        ytrain = bird_function(Xtrain_original');

        % Test Data
        X_test = zeros(2,n_sample_test);
        X_test(1,:) = (x_upperlimit - x_lowerlimit).*rand(1,n_sample_test) + x_lowerlimit;
        X_test(2,:) = (y_upperlimit - y_lowerlimit).*rand(1,n_sample_test) + y_lowerlimit;
        y_test = bird_function(X_test');

        %Points from Xtest which are in H set
        counting = X_test(:,y_test>=h);

        Xtrain = Xtrain_original;
        ytrain = bird_function(Xtrain');

        for iteration = 1:no_iterations
                
            U1_length = [U1_length1;U1_length2];
            norm_length = [norm_length1;norm_length2];

            %% Choosing next point
            %random samples to try...
            x_sample = zeros(2,1);
            x_sample(1,:) = (x_upperlimit - x_lowerlimit).*rand(1,1) + x_lowerlimit;
            x_sample(2,:) = (y_upperlimit - y_lowerlimit).*rand(1,1) + y_lowerlimit;
            
            param = default_length.*ones(size(x_sample));
            param_training = default_length.*ones(size(Xtrain));
            x_next = LSE_warped(Xtrain,ytrain,x_sample,h,param,param_training,norm_length,U1_length,value,noise,iteration,lb,ub);

            %% Sample at next point
            y_next = bird_function(x_next');
            Xtrain = [Xtrain x_next];
            ytrain = double([ytrain;y_next]);

        end
        
        %%
        %Length of Xtrain
        n_Xtrain = size(Xtrain,2); 

        param = default_length.*ones(size(X_test));
        param_training = default_length.*ones(size(Xtrain));
        
        %Build function GP
        [mu_s,std2] = build_GP(Xtrain,ytrain,X_test,param,param_training,value,noise);
        
        %Classify Test points using this GP. 
        %F1 score generated
        H_length_warped = H_length_warped + F1_score(mu_s,std2,X_test,n_sample_test,counting,h);
        
    end
    
    H_length_warped = -1*H_length_warped/(no_trials + 1);

end

 