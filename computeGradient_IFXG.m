function grad = computeGradient_IFXG(u, X, s_batch, a, b, nclass)
% computeGradient_IFXG
% Gradient of the IF-weighted XG loss with respect to beta.
%
% u       : residual matrix (m x nclass), u_ij = (X*beta)_ij - Y_ij
% X       : feature matrix (m x dim)
% s_batch : IF score vector (m x 1)
% a, b    : XG loss parameters
% nclass  : number of classes

    s_batch = s_batch(:);

    [m_batch, ~] = size(u);
    grad = zeros(size(X, 2), nclass);

    for j = 1:nclass
        for i = 1:m_batch
            s_ij = s_batch(i);
            xi   = s_ij * u(i, j);

            % XG loss derivative with respect to xi:
            % L'(xi) = a((a*xi + 1)*exp(a*xi) - 1) /
            %          (1 + a*b*xi*(exp(a*xi) - 1))^2
            exp_axi = exp(a * xi);
            num     = a * ((a*xi + 1) * exp_axi - 1);
            den     = (1 + a * b * xi * (exp_axi - 1))^2 + eps;
            dL_dxi  = num / den;

            % Chain rule because xi_ij = s_i * u_ij.
            dL_du = s_ij * dL_dxi;

            grad(:, j) = grad(:, j) + dL_du * X(i, :)';
        end
    end
end
