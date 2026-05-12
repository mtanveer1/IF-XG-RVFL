function [EVAL_Train, EVAL_Test, TrainTime, TestTime] = IXG_RVFL_Function(trainX, trainY, testX, testY, ScoreTrain, option)
% IXG_RVFL_Function
% Intuitionistic Fuzzy XG-RVFL for binary classification.
%
% trainY, testY : numeric binary labels in {0,1}
% ScoreTrain    : IF score vector for the training samples

    % Options.
    N          = option.N;
    C          = option.C;
    activation = option.activation;
    a          = option.a;   % XG loss asymmetry/steepness parameter
    b          = option.b;   % XG loss boundedness parameter
    max_iter   = getOption(option, 'max_iter', 100);
    tol        = getOption(option, 'tol', 1e-6);
    kappa0     = getOption(option, 'kappa0', 0.01);
    alpha      = getOption(option, 'alpha', 0.1);
    gamma      = getOption(option, 'gamma', 0.6);

    if numel(ScoreTrain) ~= numel(trainY)
        error('ScoreTrain must contain one IF score for each training sample.');
    end

    classes = unique(trainY);
    nclass  = numel(classes);
    if nclass ~= 2
        error('IXG_RVFL_Function expects binary labels.');
    end

    % One-hot encode training labels.
    numTrain = numel(trainY);
    dataY_train_temp = zeros(numTrain, nclass);
    for k = 1:nclass
        idx = (trainY == classes(k));
        dataY_train_temp(idx, k) = 1;
    end

    [Nsample, Nfea] = size(trainX);
    trainXrand        = trainX;
    trainYrand_onehot = dataY_train_temp;
    s_batch           = ScoreTrain(:);

    [~, idx_true] = max(trainYrand_onehot, [], 2);
    TrainLabelsBatch = classes(idx_true);

    % Hidden layer.
    tic;
    scale = 1;
    W    = (rand(Nfea, N)*2*scale - 1);
    bias = scale*rand(1, N);
    X1   = trainXrand*W + repmat(bias, Nsample, 1);
    X1   = applyActivation(X1, activation);

    % RVFL design matrix with direct links and bias.
    X = [trainXrand, X1];
    X = [X, ones(Nsample, 1)];

    % NAG-style gradient optimization with IF-weighted XG loss.
    beta = ones(size(X, 2), nclass)*0.01;
    v = zeros(size(X, 2), nclass);
    betaPrevious = inf;

    for t = 1:max_iter
        beta_look = beta + gamma*v;

        % Residual matrix of size Nsample-by-nclass.
        u = X*beta_look - trainYrand_onehot;

        grad_loss = computeGradient_IFXG(u, X, s_batch, a, b, nclass);
        gradient = beta_look + C*grad_loss;

        eta_t = kappa0 * exp(-alpha*(t - 1));
        v = gamma*v - eta_t*gradient;
        beta = beta + v;

        if norm(beta - betaPrevious, 'fro') < tol
            break;
        else
            betaPrevious = beta;
        end
    end

    TrainTime = toc;

    scoresTrain = X*beta;
    [~, idx_pred_tr] = max(scoresTrain, [], 2);
    Predict_Y_train = classes(idx_pred_tr);
    EVAL_Train = Evaluate(TrainLabelsBatch, Predict_Y_train);

    % Testing.
    tic;
    Nsample_te = size(testX, 1);
    X1_te = testX*W + repmat(bias, Nsample_te, 1);
    X1_te = applyActivation(X1_te, activation);

    X_te = [testX, X1_te];
    X_te = [X_te, ones(Nsample_te, 1)];

    rawScore_te = X_te*beta;
    [~, idx_pred_te] = max(rawScore_te, [], 2);
    Predict_Y_test = classes(idx_pred_te);

    EVAL_Test = Evaluate(testY, Predict_Y_test);
    TestTime = toc;
end

function value = getOption(option, name, defaultValue)
    if isfield(option, name)
        value = option.(name);
    else
        value = defaultValue;
    end
end
