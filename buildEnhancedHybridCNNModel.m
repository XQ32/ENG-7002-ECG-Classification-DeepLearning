function [net, lgraph] = buildEnhancedHybridCNNModel(numClasses, inputSequenceLength, varargin)
% buildEnhancedHybridCNNModel - Build enhanced hybrid CNN model (fully configurable version)
%
% This function builds a fully configurable hybrid input network for ECG heartbeat classification:
% 1. Waveform branch: Configurable fully connected layer architecture for extracting temporal features
% 2. Context branch: Configurable feature extraction layers
% 3. Simple fusion mechanism: Concatenation + configurable fully connected layers
% 4. Comprehensive regularization: Configurable Dropout, batch normalization, activation functions
%
% Inputs:
%   numClasses - Number of classification classes
%   inputSequenceLength - Length of input waveform sequence
%
% Optional parameters (Name-Value pairs):
%   'ContextDim' - Context feature dimension, default 4
%   'dl_params' - Complete deep learning parameter structure (recommended)
%   'DropoutRate' - Dropout rate, default 0.5 (backward compatibility)
%
% Outputs:
%   net - dlnetwork object for prediction (without classificationLayer)
%   lgraph - layerGraph object for training (with classificationLayer)

% Parse input parameters
p = inputParser;
addParameter(p, 'ContextDim', 4, @(x) isnumeric(x) && x >= 2);
addParameter(p, 'DropoutRate', 0.5, @(x) isnumeric(x) && x >= 0 && x <= 1);
addParameter(p, 'dl_params', [], @isstruct);
parse(p, varargin{:});

contextDim = p.Results.ContextDim;
dropoutRate = p.Results.DropoutRate;
dl_params = p.Results.dl_params;

% If dl_params is provided, use its parameters; otherwise use default values
if ~isempty(dl_params)
    fprintf('Building network using complete dl_params configuration...\n');
    
    % Extract architecture parameters from dl_params
    if isfield(dl_params, 'architecture')
        waveformLayers = dl_params.architecture.waveformBranchLayers;
        contextLayers = dl_params.architecture.contextBranchLayers;
        fusionLayers = dl_params.architecture.fusionLayers;
        commonFeatureSize = dl_params.architecture.commonFeatureSize;
    else
        % Default architecture parameters
        waveformLayers = [128, 64];
        contextLayers = [64];
        fusionLayers = [64, 32];
        commonFeatureSize = 64;
    end
    
    % Extract regularization parameters from dl_params
    if isfield(dl_params, 'regularization')
        mainDropoutRate = dl_params.regularization.dropoutRate;
        waveformDropoutRate = dl_params.regularization.waveformDropoutRate;
        fusionDropoutRate = dl_params.regularization.fusionDropoutRate;
    else
        mainDropoutRate = dropoutRate;
        waveformDropoutRate = dropoutRate * 0.5;
        fusionDropoutRate = dropoutRate;
    end
    
    % Extract activation function parameters from dl_params
    if isfield(dl_params, 'activation')
        activationType = dl_params.activation.type;
        leakyAlpha = dl_params.activation.leakyAlpha;
    else
        activationType = 'relu';
        leakyAlpha = 0.01;
    end
    
    % Extract context dimension from dl_params
    if isfield(dl_params, 'context')
        contextDim = dl_params.context.dimension;
    end
    
else
    fprintf('Building network using default parameters (backward compatibility mode)...\n');
    
    % Default architecture parameters
    waveformLayers = [128, 64];
    contextLayers = [64];
    fusionLayers = [64, 32];
    commonFeatureSize = 64;
    
    % Default regularization parameters
    mainDropoutRate = dropoutRate;
    waveformDropoutRate = dropoutRate * 0.5;
    fusionDropoutRate = dropoutRate;
    
    % Default activation function
    activationType = 'relu';
    leakyAlpha = 0.01;
end

fprintf('Building enhanced hybrid CNN model: context dimension=%d, main dropout=%.2f\n', contextDim, mainDropoutRate);

%% =================================================================
%% Helper function: Create activation layer
%% =================================================================
    function activationLayer = createActivationLayer(name)
        switch lower(activationType)
            case 'relu'
                activationLayer = reluLayer('Name', name);
            case 'leakyrelu'
                activationLayer = leakyReluLayer(leakyAlpha, 'Name', name);
            case 'elu'
                activationLayer = eluLayer('Name', name);
            otherwise
                activationLayer = reluLayer('Name', name);
        end
    end

%% =================================================================
%% Build complete network (configurable architecture)
%% =================================================================

% === Waveform branch (fully configurable) ===
waveformBranchLayers = [
    featureInputLayer(inputSequenceLength, 'Name', 'waveform_input', 'Normalization', 'none')
];

% Dynamically build fully connected layers for waveform branch
for i = 1:length(waveformLayers)
    layerName = sprintf('fc_wave_%d', i);
    waveformBranchLayers = [waveformBranchLayers
        fullyConnectedLayer(waveformLayers(i), 'Name', layerName)
        createActivationLayer(sprintf('relu_wave_%d', i))
        dropoutLayer(waveformDropoutRate, 'Name', sprintf('dropout_wave_%d', i))
    ];
end

% Final waveform feature layer
waveformBranchLayers = [waveformBranchLayers
    fullyConnectedLayer(commonFeatureSize, 'Name', 'fc_wave_features')
    createActivationLayer('relu_wave_features')
];

% Create layerGraph and add waveform branch
lgraph = layerGraph();
lgraph = addLayers(lgraph, waveformBranchLayers);

%% =================================================================
%% Add context branch (configurable)
%% =================================================================

contextBranchLayers = [
    featureInputLayer(contextDim, 'Name', 'context_input', 'Normalization', 'none')
];

% Dynamically build fully connected layers for context branch
for i = 1:length(contextLayers)
    layerName = sprintf('fc_context_%d', i);
    contextBranchLayers = [contextBranchLayers
        fullyConnectedLayer(contextLayers(i), 'Name', layerName)
        createActivationLayer(sprintf('relu_context_%d', i))
    ];
end

% Final context feature layer
contextBranchLayers = [contextBranchLayers
    fullyConnectedLayer(commonFeatureSize, 'Name', 'fc_context_features')
    createActivationLayer('relu_context_features')
];

lgraph = addLayers(lgraph, contextBranchLayers);

%% =================================================================
%% Add fusion and classification branch (configurable)
%% =================================================================

fusionBranchLayers = [
    % Feature fusion
    concatenationLayer(1, 2, 'Name', 'concat_features')
];

% Dynamically build fully connected layers after fusion
for i = 1:length(fusionLayers)
    layerName = sprintf('fc_fusion_%d', i);
    fusionBranchLayers = [fusionBranchLayers
        fullyConnectedLayer(fusionLayers(i), 'Name', layerName)
        createActivationLayer(sprintf('relu_fusion_%d', i))
        dropoutLayer(fusionDropoutRate, 'Name', sprintf('dropout_fusion_%d', i))
    ];
end

% Final classification layer
fusionBranchLayers = [fusionBranchLayers
    fullyConnectedLayer(numClasses, 'Name', 'fc_classification')
    softmaxLayer('Name', 'softmax_output')
    classificationLayer('Name', 'classification_output')
];

lgraph = addLayers(lgraph, fusionBranchLayers);

%% =================================================================
%% Establish network connections
%% =================================================================

fprintf('  Establishing network connections...\n');

% Connect waveform features to fusion layer
lgraph = connectLayers(lgraph, 'relu_wave_features', 'concat_features/in1');

% Connect context features to fusion layer
lgraph = connectLayers(lgraph, 'relu_context_features', 'concat_features/in2');

%% =================================================================
%% Network validation
%% =================================================================

% Validate network graph integrity
try
    analyzeNetwork(lgraph);
    fprintf('✓ Enhanced hybrid CNN network graph structure validation passed\n');
catch ME
    warning('NETWORK:ValidationWarning', 'Network graph validation warning: %s', ME.message);
    fprintf('Attempting to continue building, may still be usable...\n');
end

% Output network statistics
%% =================================================================
%% Output network statistics
%% =================================================================

fprintf('\n=== Enhanced hybrid CNN network construction completed (fully configurable version) ===\n');
fprintf('Network configuration:\n');
fprintf('  - Context feature dimension: %d\n', contextDim);
fprintf('  - Unified feature dimension: %d\n', commonFeatureSize);
fprintf('  - Waveform branch layers: [%s]\n', num2str(waveformLayers));
fprintf('  - Context branch layers: [%s]\n', num2str(contextLayers));
fprintf('  - Fusion layers: [%s]\n', num2str(fusionLayers));
fprintf('  - Activation function: %s\n', activationType);
fprintf('  - Main dropout rate: %.2f\n', mainDropoutRate);
fprintf('  - Waveform branch dropout: %.2f\n', waveformDropoutRate);
fprintf('  - Fusion layer dropout: %.2f\n', fusionDropoutRate);
fprintf('  - Number of classification classes: %d\n', numClasses);

%% =================================================================
%% Convert to dlnetwork (for prediction)
%% =================================================================

% To support the predict function, we need to create a version without classificationLayer
try
    % Create network for prediction (remove classificationLayer)
    lgraph_predict = removeLayers(lgraph, 'classification_output');
    net = dlnetwork(lgraph_predict);
    fprintf('✓ Network construction and conversion successful\n');
catch ME
    fprintf('❌ Network conversion failed: %s\n', ME.message);
    error('Network construction failed, please check network architecture');
end

end
