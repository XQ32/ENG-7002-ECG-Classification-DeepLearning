function lgraph = buildHybridCNNModel(numClasses, inputSequenceLength, varargin)
% buildHybridCNNModel - Build hybrid model for modern trainnet workflow
%
% This function builds a hybrid input network for ECG heartbeat classification.
% The architecture uses feature addition (Addition Layer) to fuse two branches, which is simpler and more stable than concatenation layers.
%
% 1. Waveform branch: Processes raw ECG waveform data.
% 2. Context branch: Processes context features such as RR intervals.
%
% The output feature dimensions of both branches are set to the same size (32), then fused through additionLayer.
% The final classificationLayer is necessary as it allows the trainnet function to automatically recognize this as a classification task
% and use the correct loss function (cross-entropy).
%
% Inputs:
%   numClasses - Number of classification classes
%   inputSequenceLength - Length of input waveform sequence
%
% Optional parameters (Name-Value pairs):
%   'ContextDim' - Context feature dimension, default 2
%
% Outputs:
%   lgraph - A layerGraph object compatible with trainnet

% Parse input parameters
p = inputParser;
addParameter(p, 'ContextDim', 2, @(x) isnumeric(x) && x >= 2);
parse(p, varargin{:});

contextDim = p.Results.ContextDim;

% Define unified feature dimension for output of both branches
commonFeatureSize = 32;

% --- Branch 1: Waveform processing ---
waveformBranch = [
    featureInputLayer(inputSequenceLength, 'Name', 'waveform_input', 'Normalization', 'zscore')
    
    fullyConnectedLayer(128, 'Name', 'fc1_wave')
    reluLayer('Name', 'relu1_wave')
    
    fullyConnectedLayer(64, 'Name', 'fc2_wave')
    reluLayer('Name', 'relu2_wave')
    
    % Output to unified feature dimension
    fullyConnectedLayer(commonFeatureSize, 'Name', 'fc_wave_output')
    reluLayer('Name', 'relu_wave_output')
];

% --- Branch 2: Context feature processing ---
contextBranch = [
    featureInputLayer(contextDim, 'Name', 'context_input', 'Normalization', 'zscore')

    % Output to unified feature dimension
    fullyConnectedLayer(commonFeatureSize, 'Name', 'fc_context_output')
    reluLayer('Name', 'relu_context_output')
];

% --- Main trunk: Feature fusion and classification ---
mainTrunk = [
    additionLayer(2, 'Name', 'add_features') % 2 inputs, element-wise addition
    
    reluLayer('Name', 'relu_fusion')
    dropoutLayer(0.6, 'Name', 'dropout_final')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification_output')
];

% --- Assemble network graph ---
lgraph = layerGraph();

lgraph = addLayers(lgraph, waveformBranch);
lgraph = addLayers(lgraph, contextBranch);
lgraph = addLayers(lgraph, mainTrunk);

% Connect branches to main trunk
lgraph = connectLayers(lgraph, 'relu_wave_output', 'add_features/in1');
lgraph = connectLayers(lgraph, 'relu_context_output', 'add_features/in2');

fprintf('Built a trainnet-compatible hybrid model based on feature addition.\n');
fprintf('Context feature dimension: %d\n', contextDim);

end 