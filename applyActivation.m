function X = applyActivation(X, activation)
% applyActivation
% Applies the hidden-node activation used by RVFL.
%
% activation:
%   1 sigmoid
%   2 sine
%   3 tribas
%   4 radbas
%   5 tansig
%   6 relu

    switch activation
        case 1
            X = 1 ./ (1 + exp(-X));
        case 2
            X = sin(X);
        case 3
            X = max(1 - abs(X), 0);
        case 4
            X = exp(-(X.^2));
        case 5
            X = 2 ./ (1 + exp(-2*X)) - 1;
        case 6
            X = max(0, X);
        otherwise
            error('Unsupported activation id: %s', mat2str(activation));
    end
end
