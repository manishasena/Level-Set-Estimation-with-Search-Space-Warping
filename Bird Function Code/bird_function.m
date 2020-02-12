%Mishra's Bird Function
%Function has been scaled down by 100

function scores = bird_function(x)
    
    n = size(x, 2);
    assert(n == 2, 'Bird function is only defined on a 2D space.')
    X = x(:, 1);
    Y = x(:, 2);
    
    scores = sin(X) .* exp((1 - cos(Y)).^2) + ... 
        cos(Y) .* exp((1 - sin(X)) .^ 2) + ...
        (X - Y) .^ 2;
    
    scores = scores/100;
end