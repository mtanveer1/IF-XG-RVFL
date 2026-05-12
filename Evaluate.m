function EVAL = Evaluate(ACTUAL, PREDICTED)
% Evaluate
% Computes accuracy, sensitivity, specificity, precision, F1-score, and G-mean.
% ACTUAL, PREDICTED : column vectors of labels in {0,1}, with 1 as positive.

    ACTUAL = ACTUAL(:);
    PREDICTED = PREDICTED(:);

    if numel(ACTUAL) ~= numel(PREDICTED)
        error('ACTUAL and PREDICTED must have the same number of elements.');
    end

    idx = (ACTUAL == 1);
    p   = sum(idx);
    n   = sum(~idx);
    N   = p + n;

    tp = sum(PREDICTED(idx) == 1);
    tn = sum(PREDICTED(~idx) == 0);
    fp = n - tn;

    tp_rate = safeDivide(tp, p);
    tn_rate = safeDivide(tn, n);

    accuracy    = 100 * safeDivide(tp + tn, N);
    sensitivity = 100 * tp_rate;
    specificity = 100 * tn_rate;
    precision   = 100 * safeDivide(tp, tp + fp);
    recall      = sensitivity;
    f_measure   = safeDivide(2 * precision * recall, precision + recall);
    gmean       = 100 * sqrt(tp_rate * tn_rate);

    EVAL = [accuracy sensitivity specificity precision f_measure gmean];
end

function value = safeDivide(num, den)
    if den == 0
        value = 0;
    else
        value = num / den;
    end
end
