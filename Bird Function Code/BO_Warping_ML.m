
%Using a cross-validation approach to determine the range for length scale
%to warp space to help acquisition function choose next best point to
%produce highest classification accuracy

h = 0.1;
value = 1;
noise = 0;

%True length parameters of the GP 
default_length = [1.391;1.5478]; 

initial_point = array2table([1.391,1.5478,0,0]); %Initalise at current length scale warping;

U1_length1 = optimizableVariable('U1_length1',[0,1.4],'Type','real');
U1_length2 = optimizableVariable('U1_length2',[0,1.5],'Type','real');
norm_length1 = optimizableVariable('norm_length1',[0,3],'Type','real');
norm_length2 = optimizableVariable('norm_length2',[0,3],'Type','real');

fun = @(x)bird_warping(x.U1_length1,x.U1_length2, x.norm_length1, x.norm_length2, h);

results_warping = bayesopt(fun,[U1_length1,U1_length2, norm_length1, norm_length2],'Verbose',0,...
    'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',10,'InitialX',initial_point)

%save('Warping_parameters.mat','results_warping')
