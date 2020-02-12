function [K] = kernel_warped(x1_raw,x2_raw,param1_raw,param2_raw,value)

    n_x1 = size(x1_raw,2);
    n_x2 = size(x2_raw,2);
    
    K = zeros(n_x1,n_x2);
    
    for i = 1:n_x1
        for j = 1:n_x2

            x1 = x1_raw(:,i);
            x2 = x2_raw(:,j);

            param1 = param1_raw(:,i);
            param2 = param2_raw(:,j);

            Z1 = param1.^2.*eye(size(x1_raw,1));
            Z2 = param2.^2.*eye(size(x2_raw,1));

            exp_term = exp(-(0.5)*(x1-x2)'*((Z1+Z2)/2)^(-1)*(x1-x2));
            K(i,j) = value^2 * det(Z1)^(1/4) * det(Z2)^(1/4) * det((Z1+Z2)/2)^(-1/2) * exp_term;

        end
    end

end