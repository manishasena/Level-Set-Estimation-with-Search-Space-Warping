
%Mishra's Bird Function Experiment Code
%Threshold set to 10 (after normalisation h = 0.1)

close all

%% Input Search Space %%%
x_points = 41;
y_points = 41;
x_lowerlimit = -4;
y_lowerlimit = -4;
x_upperlimit = 4;
y_upperlimit = 4;

%Test of space (calculate points in H set)
total = zeros(1,x_points*y_points);
n = 0;
t = 0;
counting = zeros(2,5);

%Level 
h = 0.1;

for i = linspace(x_lowerlimit,x_upperlimit,x_points)
    
    for j = linspace(y_lowerlimit,y_upperlimit,y_points)
        
        n = n + 1;
        total(n) = bird_function([i;j]'); %Function has been normalised to range between -1 and 1
    
        if total(n) >= h

            t = t + 1;
            counting(:,t) = [i;j];

        end

    end
    
end

total = reshape(total,x_points,y_points);
[U,V] = meshgrid(linspace(y_lowerlimit,y_upperlimit,x_points), linspace(x_lowerlimit,x_upperlimit,y_points));
x = [V(:),U(:)]';

%% Display true function
figure(1)
scatter3(0,0,2,'w')
hold on
surf(U,V,h*ones(size(total)),-1*ones(size(total)))
shading interp
surf(U,V,total)
hold off

%% Parameters %%%
%Kernel parameter
value = 1;

%No Noise
noise = 0.00; 

%Number of random trials
no_trials = 50;

%Number of iterations per trial
no_iterations = 100;

%Length Parameter of true GP
default_length = [1.391,1.5478]';

%Warping Parameters
%Pre-tuned offline
% To access warping parameter from optimiser
%warping_parameters = table2array(results_warping.XAtMinObjective);

% Pre-tuned warping parameters 
warping_parameters = [1.318663048	0.002768616	0.01708057	0.383851688];
L1 = warping_parameters(1:2)'; 
L0 = warping_parameters(3:4)'; 

%Classification Accuracy No warping
%Matrix to store F1 score in each trial and iteration
H_length = zeros(no_trials,no_iterations);

%Classification Accuracy Input warping
%Matrix to store F1 score in each trial and iteration
H_length_warped = zeros(no_trials,no_iterations);

%Test points to classify
n_test = 500;
r_test = randi([1 n],1,n_test);
X_test = x(:,r_test);
y_test = bird_function(X_test');

%Pre-classify test points (to check against)
%Two classes, either equal above h (super-level), or not
classification_test = y_test > h;
counting = X_test(:,classification_test);

%Begin experiment
for trial = 1:no_trials
    
    %Display Trial Number
    disp('Trial number:')
    disp(trial)
    
    %Initialise 3 random points 
    r = randi([1 n],1,3);
    Xtrain = x(:,r);
    ytrain = bird_function(Xtrain');
  
    %% Without Warping Experiment
    for iteration = 1:no_iterations 
        
        disp(iteration)
        
        %Length parameters of true function
        param = default_length.*ones(size(X_test));
        param_training = default_length.*ones(size(Xtrain));
        
        %Build function GP
        [mu_s,std2] = build_GP(Xtrain,ytrain,X_test,param,param_training,value,noise);
        
        %Classify Test points using this GP. 
        %F1 score generated
        H_length(trial,iteration) = F1_score(mu_s,std2,X_test,n_test,counting,h);

        % Choosing next point to sample
        %Intialisation
        x_sample = zeros(2,1);
        x_sample(1,:) = (x_upperlimit - x_lowerlimit).*rand(1,1) + x_lowerlimit;
        x_sample(2,:) = (y_upperlimit - y_lowerlimit).*rand(1,1) + y_lowerlimit;

        param = default_length.*ones(size(x_sample));
        
        %Lower and Upper limits
        lb = [x_lowerlimit,y_lowerlimit];
        ub = [x_upperlimit,y_upperlimit];
    
        %Samples from LSE acquisition function and retrieves next best sample point.
        x_next = LSE(Xtrain,ytrain,x_sample,h,param,param_training,value,noise,iteration,lb,ub);
        
        %Sample at this point and add to Training data
        y_next = bird_function(x_next');
        Xtrain = [Xtrain x_next];
        ytrain = double([ytrain;y_next]);
        
    end
    
    %% With Warping Experiment
    
    %Using same initial 3 training points.
    Xtrain = x(:,r);
    ytrain = bird_function(Xtrain');

    for iteration = 1:no_iterations

        disp(iteration)
        
        %Length parameters of true function
        param = default_length.*ones(size(X_test));
        param_training = default_length.*ones(size(Xtrain));
        
        %Build function GP
        [mu_s,std2] = build_GP(Xtrain,ytrain,X_test,param,param_training,value,noise);
        
        %Classify Test points using this GP. 
        %F1 score generated
        H_length_warped(trial,iteration) = F1_score(mu_s,std2,X_test,n_test,counting,h);
        
        % Choosing next point to sample
        %Intialisation
        x_sample = zeros(2,1);
        x_sample(1,:) = (x_upperlimit - x_lowerlimit).*rand(1,1) + x_lowerlimit;
        x_sample(2,:) = (y_upperlimit - y_lowerlimit).*rand(1,1) + y_lowerlimit;

        param = default_length.*ones(size(x_sample));
        
        %Lower and Upper limits
        lb = [x_lowerlimit,y_lowerlimit];
        ub = [x_upperlimit,y_upperlimit];
        
        %Samples from LSE acquisition function with input space warping
        %and retrieves next sample point.
        x_next = LSE_warped(Xtrain,ytrain,x_sample,h,param,param_training,L0,L1,value,noise,iteration,lb,ub);
        
        %Sample at this point and add to Training data
        y_next = bird_function(x_next');
        Xtrain = [Xtrain x_next];
        ytrain = double([ytrain;y_next]); 
   
    end

end
%% Plotting results.

%Average
H_length_avg = mean(H_length);
H_length_warped_avg = mean(H_length_warped);

%Standard Deviation
H_length_std = std(H_length)/sqrt(no_trials);
H_length_warped_std = std(H_length_warped)/sqrt(no_trials);

increment = 2;

if no_trials > 1
    figure()
    
    iterations = [1,5:increment:size(H_length_avg,2)];
    errorbar(iterations,H_length_avg(iterations),H_length_std(iterations),'b','LineWidth',1.5)
    hold on
    errorbar(iterations,H_length_warped_avg(iterations),H_length_warped_std(iterations),'r','LineWidth',1.5)
    xlabel('Iterations')
    ylabel('F1 Score')
    %title('Bird Function: Threshold = 0.1')
    legend({'LSE','LSE warped'},'Location','southeast')
    ylim([0,1])
    hold off
    %print -depsc Birdfunction_results.eps

else
    figure()
    plot(H_length)
    hold on
    plot(H_length_warped,'r')
    legend({'LSE','LSE warped'},'Location','southeast')
    ylim([0,1])
    hold off
    %print -depsc Birdfunction_results.eps
end



%% Save results
%save('H_length.mat','H_length')
%save('H_length_warped.mat','H_length_warped')