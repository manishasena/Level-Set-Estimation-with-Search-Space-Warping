function [score] = F1_score(mu_s,std2,X_test,n_sample_test,counting,h)

    Upper  = mu_s + 2*std2;
    Lower  = mu_s - 2*std2;

    % Sets
    U1 = [];
    H = [];
    L = [];

    for k = 1:n_sample_test
        if Lower(k) >= h
            H = [H,X_test(:,k)];
        elseif Upper(k) < h
            L = [L, X_test(:,k)];
        else
            U1 = [U1,X_test(:,k)];
        end
    end

    %% F1 score for H
    if size(H,2) > 0
        TP = sum(ismember(H',counting','rows'));
        FP = size(H,2) - TP;
        FN = sum(~ismember(counting',H','rows'));

        precision = TP/(TP + FP);
        recall = TP/(TP + FN);

        if (precision + recall) == 0
            score = 0;
        else
            score = 2*precision*recall/(precision + recall);  
        end

    else
        score = 0;
    end

end