% Main.m
% Demo script for IF-XG-RVFL.
%
% This script runs IF-XG-RVFL with fixed hyperparameter setting.
% The full hyperparameter tuning and cross-validation protocol used in the
% paper is described in the experimental setup of the manuscript.

clear;
clc;

rng(1, 'twister');

% -------------------------------------------------------------------------
% Paths
% -------------------------------------------------------------------------
% Put your .mat datasets in this folder. Each .mat file should contain a
% numeric matrix with samples in rows, features in columns, and the class
% label in the last column.
datasetDir = fullfile(pwd, 'data');       % PUT_DATASET_PATH_HERE
resultsDir = fullfile(pwd, 'results');    % PUT_RESULTS_SAVING_PATH_HERE

if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

datasetFiles = dir(fullfile(datasetDir, '*.mat'));
if isempty(datasetFiles)
    error('No .mat dataset files found in %s. Put datasets in the data folder or update datasetDir.', datasetDir);
end

% -------------------------------------------------------------------------
% Fixed hyperparameters for a clean demo.
% -------------------------------------------------------------------------
% For the complete tuning grid and cross-validation protocol, refer to the
% experimental setup described in the paper. The following values provide a
% simple starting point.
option.C          = 1;
option.N          = 100;
option.activation = 1;      % 1 sigmoid, 2 sine, 3 tribas, 4 radbas, 5 tansig, 6 relu
option.a          = 1;      % XG loss asymmetry/steepness parameter
option.b          = 1;      % XG loss boundedness parameter
option.mew        = 1;      % RBF/IF kernel width parameter
option.trainRatio = 0.80;

% NAG optimization parameters used as fixed reproducible settings.
option.max_iter = 100;
option.tol      = 1e-6;
option.kappa0   = 0.01;    % initial learning rate
option.alpha    = 0.1;     % learning-rate decay factor
option.gamma    = 0.6;     % momentum coefficient

% -------------------------------------------------------------------------
% Tuning grid used in the paper.
% -------------------------------------------------------------------------
% C_grid          = 10.^(-5:1:5);
% N_grid          = 3:20:203;
% activation_grid = 1:6;
% a_grid          = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2];
% b_grid          = 0.5:0.5:2;
% mew_grid        = 2.^(-5:1:5);

resultFile = fullfile(resultsDir, 'IF_XG_RVFL_results.tsv');
fid = fopen(resultFile, 'w');
if fid < 0
    error('Unable to create result file: %s', resultFile);
end
cleanupObj = onCleanup(@() fclose(fid));

fprintf(fid, 'Dataset\tTrainAccuracy\tTestAccuracy\tSensitivity\tSpecificity\tPrecision\tF1Score\tGMean\tTrainTime\tTestTime\tC\tN\tActivation\ta\tb\tmew\n');

for dataIdx = 1:numel(datasetFiles)
    datasetName = datasetFiles(dataIdx).name;
    fprintf('Processing %s (%d/%d)\n', datasetName, dataIdx, numel(datasetFiles));

    datasetPath = fullfile(datasetFiles(dataIdx).folder, datasetName);
    allData = loadDatasetMatrix(datasetPath);

    X = allData(:, 1:end-1);
    y = allData(:, end);

    [y01, originalLabels] = mapBinaryLabels(y);
    X = double(X);

    [trainIdx, testIdx] = stratifiedHoldout(y01, option.trainRatio);

    trainX = X(trainIdx, :);
    testX  = X(testIdx, :);
    trainY = y01(trainIdx);
    testY  = y01(testIdx);

    % Standardize using training statistics only.
    mu = mean(trainX, 1);
    sigma = std(trainX, 0, 1);
    sigma(sigma == 0) = 1;
    trainX = (trainX - mu) ./ sigma;
    testX  = (testX - mu) ./ sigma;

    trainDataForScore = [trainX, trainY];
    scoreTrain = score_fun(trainDataForScore, option.mew);

    [evalTrain, evalTest, trainTime, testTime] = ...
        IXG_RVFL_Function(trainX, trainY, testX, testY, scoreTrain, option);

    fprintf(fid, '%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.6f\t%.6f\t%.6g\t%d\t%d\t%.6g\t%.6g\t%.6g\n', ...
        datasetName, evalTrain(1), evalTest(1), evalTest(2), evalTest(3), ...
        evalTest(4), evalTest(5), evalTest(6), trainTime, testTime, ...
        option.C, option.N, option.activation, option.a, option.b, option.mew);

    fprintf('  Original labels mapped to 0/1: %s -> 0, %s -> 1\n', ...
        mat2str(originalLabels(1)), mat2str(originalLabels(2)));
    fprintf('  Test accuracy: %.4f%%, F1-score: %.4f%%\n', evalTest(1), evalTest(5));
end

fprintf('Results saved to: %s\n', resultFile);

function data = loadDatasetMatrix(datasetPath)
    raw = load(datasetPath);
    names = fieldnames(raw);
    data = [];

    for i = 1:numel(names)
        candidate = raw.(names{i});
        if isnumeric(candidate) && ismatrix(candidate) && size(candidate, 1) > 1 && size(candidate, 2) > 1
            data = candidate;
            break;
        end
    end

    if isempty(data)
        error('Dataset file %s does not contain a valid numeric matrix.', datasetPath);
    end

    data = double(data);
    data = data(all(isfinite(data), 2), :);
end

function [y01, labels] = mapBinaryLabels(y)
    labels = unique(y(:));
    labels = sort(labels);

    if numel(labels) ~= 2
        error('IF-XG-RVFL demo expects a binary classification dataset. Found %d classes.', numel(labels));
    end

    y01 = double(y(:) == labels(2));
end

function [trainIdx, testIdx] = stratifiedHoldout(y, trainRatio)
    trainIdx = false(size(y));
    testIdx = false(size(y));
    classes = unique(y(:))';

    for cls = classes
        idx = find(y == cls);
        if numel(idx) < 2
            error('Each class must contain at least two samples for stratified holdout splitting.');
        end
        idx = idx(randperm(numel(idx)));
        nTrain = max(1, floor(trainRatio * numel(idx)));
        nTrain = min(nTrain, numel(idx) - 1);

        trainIdx(idx(1:nTrain)) = true;
        testIdx(idx(nTrain+1:end)) = true;
    end
end
