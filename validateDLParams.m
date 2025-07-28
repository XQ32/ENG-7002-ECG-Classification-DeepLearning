function [dl_params, isValid, messages] = validateDLParams(dl_params)
%validateDLParams Validate and standardize the deep learning parameter structure
%
% This function checks the integrity and validity of the dl_params structure and provides default values
%
% Input:
%   dl_params - Deep learning parameter structure
%
% Output:
%   dl_params - Parameter structure after validation and supplementation with default values
%   isValid - Logical value, indicating whether the parameters are valid
%   messages - String array, containing validation information and warnings

messages = {};
isValid = true;

fprintf('Validating deep learning parameter configuration...\n');

%% Basic Parameter Validation
if ~isfield(dl_params, 'standardLength') || dl_params.standardLength <= 0
    dl_params.standardLength = 288;
    messages{end+1} = 'Warning: standardLength is invalid, set to default value 288';
end

if ~isfield(dl_params, 'networkType')
    dl_params.networkType = 'enhanced';
    messages{end+1} = 'Info: networkType set to default value enhanced';
end

%% Architecture Parameter Validation and Default Values
if ~isfield(dl_params, 'architecture')
    dl_params.architecture = struct();
end

if ~isfield(dl_params.architecture, 'waveformBranchLayers')
    dl_params.architecture.waveformBranchLayers = [128, 64];
end

if ~isfield(dl_params.architecture, 'contextBranchLayers')
    dl_params.architecture.contextBranchLayers = [64];
end

if ~isfield(dl_params.architecture, 'fusionLayers')
    dl_params.architecture.fusionLayers = [64, 32];
end

if ~isfield(dl_params.architecture, 'commonFeatureSize')
    dl_params.architecture.commonFeatureSize = 64;
end

% Validate architecture parameters
if any(dl_params.architecture.waveformBranchLayers <= 0)
    isValid = false;
    messages{end+1} = 'Error: Waveform branch layer sizes must be positive numbers';
end

if any(dl_params.architecture.contextBranchLayers <= 0)
    isValid = false;
    messages{end+1} = 'Error: Context branch layer sizes must be positive numbers';
end

if any(dl_params.architecture.fusionLayers <= 0)
    isValid = false;
    messages{end+1} = 'Error: Fusion layer sizes must be positive numbers';
end

%% Regularization Parameter Validation and Default Values
if ~isfield(dl_params, 'regularization')
    dl_params.regularization = struct();
end

if ~isfield(dl_params.regularization, 'dropoutRate')
    dl_params.regularization.dropoutRate = 0.5;
end

if ~isfield(dl_params.regularization, 'waveformDropoutRate')
    dl_params.regularization.waveformDropoutRate = 0.25;
end

if ~isfield(dl_params.regularization, 'fusionDropoutRate')
    dl_params.regularization.fusionDropoutRate = 0.5;
end

% Validate Dropout rates
dropoutFields = {'dropoutRate', 'waveformDropoutRate', 'fusionDropoutRate'};
for i = 1:length(dropoutFields)
    field = dropoutFields{i};
    value = dl_params.regularization.(field);
    if value < 0 || value > 1
        isValid = false;
        messages{end+1} = sprintf('Error: %s must be between 0 and 1', field);
    end
end

%% Activation Function Parameter Validation and Default Values
if ~isfield(dl_params, 'activation')
    dl_params.activation = struct();
end

if ~isfield(dl_params.activation, 'type')
    dl_params.activation.type = 'relu';
end

if ~isfield(dl_params.activation, 'leakyAlpha')
    dl_params.activation.leakyAlpha = 0.01;
end

% Validate activation function type
validActivations = {'relu', 'leakyrelu', 'elu'};
if ~ismember(lower(dl_params.activation.type), validActivations)
    messages{end+1} = sprintf('Warning: Unknown activation function type "%s", set to relu', dl_params.activation.type);
    dl_params.activation.type = 'relu';
end

%% Training Parameter Validation and Default Values
if ~isfield(dl_params, 'training')
    dl_params.training = struct();
end

if ~isfield(dl_params.training, 'optimizer')
    dl_params.training.optimizer = 'adam';
end

if ~isfield(dl_params.training, 'initialLearningRate')
    dl_params.training.initialLearningRate = 2e-3;
end

if ~isfield(dl_params.training, 'miniBatchSize')
    dl_params.training.miniBatchSize = 256;
end

if ~isfield(dl_params.training, 'maxEpochs')
    dl_params.training.maxEpochs = 20;
end

if ~isfield(dl_params.training, 'validationSplit')
    dl_params.training.validationSplit = 0.1;
end

% Validate training parameters
if dl_params.training.initialLearningRate <= 0 || dl_params.training.initialLearningRate > 1
    isValid = false;
    messages{end+1} = 'Error: Learning rate must be between 0 and 1';
end

if dl_params.training.miniBatchSize <= 0 || mod(dl_params.training.miniBatchSize, 1) ~= 0
    isValid = false;
    messages{end+1} = 'Error: Mini batch size must be a positive integer';
end

if dl_params.training.maxEpochs <= 0 || mod(dl_params.training.maxEpochs, 1) ~= 0
    isValid = false;
    messages{end+1} = 'Error: Max epochs must be a positive integer';
end

if dl_params.training.validationSplit <= 0 || dl_params.training.validationSplit >= 1
    isValid = false;
    messages{end+1} = 'Error: Validation split must be between 0 and 1';
end

%% Classification Parameter Validation and Default Values
if ~isfield(dl_params, 'classification')
    dl_params.classification = struct();
end

if ~isfield(dl_params.classification, 'task')
    dl_params.classification.task = 'binary';
end

if ~isfield(dl_params.classification, 'classNames')
    dl_params.classification.classNames = {'Other', 'PVC'};
end

if ~isfield(dl_params.classification, 'threshold')
    dl_params.classification.threshold = 0.2;
end

% Validate classification parameters
validTasks = {'binary', 'multiclass'};
if ~ismember(lower(dl_params.classification.task), validTasks)
    isValid = false;
    messages{end+1} = 'Error: Classification task must be binary or multiclass';
end

if dl_params.classification.threshold <= 0 || dl_params.classification.threshold >= 1
    isValid = false;
    messages{end+1} = 'Error: Classification threshold must be between 0 and 1';
end

%% Context Parameter Validation and Default Values
if ~isfield(dl_params, 'context')
    dl_params.context = struct();
end

if ~isfield(dl_params.context, 'enhanced')
    dl_params.context.enhanced = true;
end

if ~isfield(dl_params.context, 'dimension')
    dl_params.context.dimension = 4;
end

if dl_params.context.dimension < 2
    isValid = false;
    messages{end+1} = 'Error: Context feature dimension must be at least 2';
end

%% Preprocessing Parameter Validation and Default Values
if ~isfield(dl_params, 'preprocessing')
    dl_params.preprocessing = struct();
end

if ~isfield(dl_params.preprocessing, 'normalizationMethod')
    dl_params.preprocessing.normalizationMethod = 'zscore';
end

if ~isfield(dl_params.preprocessing, 'enableQualityCheck')
    dl_params.preprocessing.enableQualityCheck = true;
end

%% Output Validation Results
fprintf('Parameter validation complete:\n');
if isValid
    fprintf('✓ All parameters are valid\n');
else
    fprintf('❌ Parameter validation failed, there are errors\n');
end

if ~isempty(messages)
    fprintf('Validation messages:\n');
    for i = 1:length(messages)
        fprintf('  %s\n', messages{i});
    end
end

end
