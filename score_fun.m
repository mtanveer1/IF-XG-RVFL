function S = score_fun(A, mew)
% score_fun
% Computes intuitionistic fuzzy credibility scores for binary data.
%
% A   : matrix [features, label], where the final column contains labels
%       encoded as 0 and 1.
% mew : Gaussian/RBF kernel width parameter.

    if mew <= 0
        error('The RBF kernel width parameter mew must be positive.');
    end

    X = A(:, 1:end-1);
    y = A(:, end);
    [no_input, ~] = size(X);

    classes = unique(y(:));
    if numel(classes) ~= 2
        error('score_fun expects a binary dataset with exactly two classes.');
    end

    positiveClass = 1;
    if ~any(classes == positiveClass)
        positiveClass = classes(end);
    end

    X_pos = X(y == positiveClass, :);
    X_neg = X(y ~= positiveClass, :);

    if isempty(X_pos) || isempty(X_neg)
        error('Both classes must be present when computing IF scores.');
    end

    K_pos = rbfKernel(X_pos, X_pos, mew);
    K_neg = rbfKernel(X_neg, X_neg, mew);
    K_all = rbfKernel(X, X, mew);

    % Class-wise radii in the induced RBF kernel space.
    radius_pos = sqrt(max(0, 1 - 2*mean(K_pos, 2) + mean(K_pos(:))));
    radius_neg = sqrt(max(0, 1 - 2*mean(K_neg, 2) + mean(K_neg(:))));

    radius_max_pos = max(radius_pos);
    radius_max_neg = max(radius_neg);
    neighborRadius = max(radius_max_pos, radius_max_neg);

    % Membership: samples closer to their class center receive higher values.
    mem = zeros(no_input, 1);
    posCounter = 1;
    negCounter = 1;
    for i = 1:no_input
        if y(i) == positiveClass
            mem(i) = 1 - radius_pos(posCounter)/(radius_max_pos + 1e-4);
            posCounter = posCounter + 1;
        else
            mem(i) = 1 - radius_neg(negCounter)/(radius_max_neg + 1e-4);
            negCounter = negCounter + 1;
        end
    end
    mem = min(max(mem, 0), 1);

    % Nonmembership: fraction of local neighbors from the opposite class.
    kernelDistance = sqrt(max(0, 2*(1 - K_all)));
    ro = zeros(no_input, 1);
    for i = 1:no_input
        neighborIdx = kernelDistance(i, :)' <= neighborRadius;
        numNeighbors = sum(neighborIdx);

        if numNeighbors == 0
            ro(i) = 0;
        else
            ro(i) = sum(y(neighborIdx) ~= y(i)) / numNeighbors;
        end
    end

    nonmembership = (1 - mem) .* ro;

    % Intuitionistic fuzzy credibility score.
    S = zeros(no_input, 1);
    for i = 1:no_input
        if nonmembership(i) == 0
            S(i) = mem(i);
        elseif mem(i) <= nonmembership(i)
            S(i) = 0;
        else
            S(i) = (1 - nonmembership(i)) / (2 - mem(i) - nonmembership(i));
        end
    end
    S = min(max(S, 0), 1);
end

function K = rbfKernel(X1, X2, mew)
    D2 = squaredDistanceMatrix(X1, X2);
    K = exp(-D2/(mew^2));
end

function D2 = squaredDistanceMatrix(X1, X2)
    X1norm = sum(X1.^2, 2);
    X2norm = sum(X2.^2, 2)';
    D2 = X1norm - 2*(X1*X2') + X2norm;
    D2 = max(D2, 0);
end
