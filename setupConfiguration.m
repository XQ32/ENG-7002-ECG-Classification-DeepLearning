function [db_allocation, mat_files_info, results_dir, dl_params] = setupConfiguration()
%setupConfiguration Set project configuration parameters
%   This function defines database allocation, MAT file information, results directory and deep learning parameters.
%
%   Output:
%       db_allocation   - Structure specifying databases for training and testing
%       mat_files_info  - Cell array containing detailed information for each database file
%       results_dir     - String, directory storing .mat files
%       dl_params       - Structure, deep learning model configuration parameters

%% Parameter Settings

% Database allocation configuration - easily modifiable parameters for specifying which databases to use for training/testing
db_allocation = struct();

% Databases for training
db_allocation.training = {'MIT_data','mit_bih_supraventricular_arrhythmia_database_1_0_0_data', 'leipzig_heart_center_ecg_database_arrhythmias_in_children_and_p', ...
    'european_st_t_database_1_0_0_data', 'mit_bih_long_term_ecg_database_1_0_0_data', 'cu_ventricular_tachyarrhythmia_database_1_0_0_data', ...
};

% Databases for testing
db_allocation.testing = {'INCART_data'};

% MAT database file list (assumed to be in 'results' folder)
mat_files_info = {
    struct('fileName', 'MIT_data.mat', 'structName', 'MIT_data', 'powerLineFreq', 60), ...
    struct('fileName', 'INCART_data.mat', 'structName', 'INCART_data', 'powerLineFreq', 50), ...
    struct('fileName', 'leipzig-heart-center-ecg-database-arrhythmias-in-children-and-patients-with-congenital-heart-disease-1.0.0_data.mat', 'structName', 'leipzig_heart_center_ecg_database_arrhythmias_in_children_and_p', 'powerLineFreq', 50), ...
    struct('fileName', 'mit-bih-supraventricular-arrhythmia-database-1.0.0_data.mat', 'structName', 'mit_bih_supraventricular_arrhythmia_database_1_0_0_data', 'powerLineFreq', 60), ...
    struct('fileName', 'european-st-t-database-1.0.0_data.mat', 'structName', 'european_st_t_database_1_0_0_data', 'powerLineFreq', 50), ...
    struct('fileName', 'cu-ventricular-tachyarrhythmia-database-1.0.0_data.mat', 'structName', 'cu_ventricular_tachyarrhythmia_database_1_0_0_data', 'powerLineFreq', 60), ...
    struct('fileName', 'mit-bih-long-term-ecg-database-1.0.0_data.mat', 'structName', 'mit_bih_long_term_ecg_database_1_0_0_data', 'powerLineFreq', 60), ...

    % struct('fileName', 'long-term-af-database-1.0.0_data.mat', 'structName', 'long_term_af_database_1_0_0_data', 'powerLineFreq', 60), ...
    % struct('fileName', 'paroxysmal-atrial-fibrillation-events-detection_data.mat', 'structName', 'paroxysmal_atrial_fibrillation_events_detection_data', 'powerLineFreq', 50), ...
    % struct('fileName', 'mit-bih-malignant-ventricular-ectopy-database-1.0.0_data.mat', 'structName', 'mit_bih_malignant_ventricular_ectopy_database_1_0_0_data', 'powerLineFreq', 60), ...
    % struct('fileName', 'sudden-cardiac-death-holter-database-1.0.0_data.mat', 'structName', 'sudden_cardiac_death_holter_database_1_0_0_data', 'powerLineFreq', 60), ...
    % struct('fileName', 'shdb-af-a-japanese-holter-ecg-database-of-atrial-fibrillation-1.0.1_data.mat', 'structName', 'shdb_af_a_japanese_holter_ecg_database_of_atrial_fibrillation_1_0_1_data', 'powerLineFreq', 50), ...
    % struct('fileName', 'recordings-excluded-from-the-nsr-db-1.0.0_data.mat', 'structName', 'recordings_excluded_from_the_nsr_db_1_0_0_data', 'powerLineFreq', 60), ...

};
results_dir = 'results'; % Directory where .mat files are located

%% ========================================================================
%  Deep Learning Network Configuration Parameters - Complete Configuration System
%  ========================================================================
fprintf('Setting up deep learning network configuration...\n');

%% Basic Configuration
dl_params.standardLength = 288; % Restore to standard length to reduce computation

%% Network Architecture Selection
% 'simple'     - Use original simple hybrid CNN model
% 'enhanced'   - Use improved enhanced hybrid CNN model (recommended)
% 'multimodal' - Use new multi-modal attention network (complex scenarios)
dl_params.networkType = 'enhanced';

%% Enhanced CNN Model Detailed Architecture Parameters
% === Network Depth and Width Parameters ===
dl_params.architecture.waveformBranchLayers = [256, 128, 64]; % Reduce layers and width to accelerate training
dl_params.architecture.contextBranchLayers = [128, 64];       % Reduce context branch complexity
dl_params.architecture.fusionLayers = [128, 64, 32];         % Simplify fusion layer
dl_params.architecture.commonFeatureSize = 64;               % Lower feature dimension to improve speed

% === Activation Function Configuration ===
dl_params.activation.type = 'relu';                      % Activation function type: 'relu', 'leakyrelu', 'elu'
dl_params.activation.leakyAlpha = 0.01;                  % Alpha parameter for LeakyReLU

% === Regularization Parameters ===
dl_params.regularization.dropoutRate = 0.4;             % Moderately increase main dropout to improve generalization
dl_params.regularization.waveformDropoutRate = 0.2;     % Lower dropout for waveform branch to preserve features
dl_params.regularization.contextDropoutRate = 0.15;     % Moderate dropout for context branch
dl_params.regularization.fusionDropoutRate = 0.5;       % Higher dropout for fusion layer to prevent overfitting

% === Batch Normalization Parameters ===
dl_params.batchNorm.enable = false;                     % Enable batch normalization
dl_params.batchNorm.momentum = 0.9;                     % Batch normalization momentum
dl_params.batchNorm.epsilon = 1e-5;                     % Batch normalization epsilon

%% Training Parameter Configuration
% === Optimizer Parameters ===
dl_params.training.optimizer = 'adam';                  % Optimizer: 'adam', 'sgdm', 'rmsprop'
dl_params.training.initialLearningRate = 1e-3;          % Increase learning rate to accelerate convergence
dl_params.training.miniBatchSize = 128;                 % Increase batch size to improve training efficiency
dl_params.training.maxEpochs = 15;                      % Further reduce epochs

% === Learning Rate Scheduling ===
dl_params.training.learnRateSchedule = 'none';          % Learning rate schedule: 'none', 'piecewise', 'exponential'
dl_params.training.learnRateDropFactor = 0.1;           % Learning rate decay factor
dl_params.training.learnRateDropPeriod = 10;            % Learning rate decay period

% === Validation and Early Stopping ===
dl_params.training.validationSplit = 0.05;              % Reduce validation set ratio to accelerate training
dl_params.training.enableEarlyStopping = true;          % Enable early stopping to avoid overtraining
dl_params.training.patience = 3;                        % Lower patience value for faster stopping

%% Classification and Evaluation Parameters
% === Classification Task Configuration ===
dl_params.classification.task = 'binary';               % Classification task: 'binary', 'multiclass'
dl_params.classification.classNames = {'Other', 'PVC'}; % Class names
dl_params.classification.threshold = 0.08;               % Lower threshold to improve recall to 90%+

% === Class Imbalance Handling ===
dl_params.classBalance.method = 'weights';              % Use weights instead of downsampling to preserve more data
dl_params.classBalance.maxRatio = 2.0;                  % Slightly increase ratio to support recall improvement
dl_params.classBalance.classWeights = [1.0, 4.0];      % Moderately increase PVC weight to improve recall

%% Data Preprocessing Configuration
% === Context Features ===
dl_params.context.enhanced = true;                      % Use enhanced 4-dimensional context features
dl_params.context.dimension = 4;                        % Context feature dimension

% === Data Normalization ===
dl_params.preprocessing.normalizationMethod = 'zscore'; % Normalization: 'zscore', 'minmax', 'robust'
dl_params.preprocessing.enableQualityCheck = true;      % Enable data quality detection

%% Advanced Feature Configuration (for extensibility)
% === Multi-modal and Attention Mechanism (reserved) ===
dl_params.advanced.enableMultiScale = false;           % Enable multi-scale feature extraction
dl_params.advanced.enableAttention = false;            % Enable attention mechanism
dl_params.advanced.attentionHeads = 4;                 % Number of attention heads

% === Model Complexity (reserved) ===
dl_params.advanced.modelComplexity = 'moderate';       % Model complexity: 'simple', 'moderate', 'complex'

%% Validation and Debug Configuration
dl_params.debug.enableValidation = true;               % Enable data validation
dl_params.debug.verbose = true;                        % Verbose output
dl_params.debug.plotTraining = true;                   % Display training progress plots

%% Parameter Validation
fprintf('\n--- Parameter Validation ---\n');
[dl_params, isValid, ~] = validateDLParams(dl_params);

if ~isValid
    error('Deep learning parameter validation failed, please check configuration');
end

%% Configuration Information Display
fprintf('\n=== Deep Learning Configuration Overview ===\n');
fprintf('Network Architecture: %s\n', dl_params.networkType);

% Display Enhanced CNN architecture parameters
fprintf('\n--- Enhanced CNN Architecture Parameters ---\n');
fprintf('Waveform Branch Layers: [%s]\n', num2str(dl_params.architecture.waveformBranchLayers));
fprintf('Context Branch Layers: [%s]\n', num2str(dl_params.architecture.contextBranchLayers));
fprintf('Fusion Layers: [%s]\n', num2str(dl_params.architecture.fusionLayers));
fprintf('Unified Feature Dimension: %d\n', dl_params.architecture.commonFeatureSize);

% Display regularization parameters
fprintf('\n--- Regularization Configuration ---\n');
fprintf('Main Dropout Rate: %.2f\n', dl_params.regularization.dropoutRate);
fprintf('Waveform Branch Dropout: %.2f\n', dl_params.regularization.waveformDropoutRate);
fprintf('Fusion Layer Dropout: %.2f\n', dl_params.regularization.fusionDropoutRate);
fprintf('Activation Function: %s\n', dl_params.activation.type);

% Display training parameters
fprintf('\n--- Training Configuration ---\n');
fprintf('Optimizer: %s\n', dl_params.training.optimizer);
fprintf('Learning Rate: %.1e\n', dl_params.training.initialLearningRate);
fprintf('Batch Size: %d\n', dl_params.training.miniBatchSize);
fprintf('Maximum Epochs: %d\n', dl_params.training.maxEpochs);
fprintf('Validation Split: %.1f\n', dl_params.training.validationSplit);

% Display classification configuration
fprintf('\n--- Classification Configuration ---\n');
fprintf('Task Type: %s (%d classes)\n', dl_params.classification.task, length(dl_params.classification.classNames));
fprintf('Target Classes: %s\n', strjoin(dl_params.classification.classNames, ', '));
fprintf('Classification Threshold: %.2f\n', dl_params.classification.threshold);
fprintf('Class Balance: %s (Max Ratio: %d)\n', dl_params.classBalance.method, dl_params.classBalance.maxRatio);

% Display data preprocessing
fprintf('\n--- Data Preprocessing ---\n');
fprintf('Standard Length: %d sample points\n', dl_params.standardLength);
if dl_params.context.enhanced
    context_type = 'Enhanced Features';
else
    context_type = 'Basic Features';
end
fprintf('Context Features: %s (%d-dim)\n', context_type, dl_params.context.dimension);
fprintf('Normalization Method: %s\n', dl_params.preprocessing.normalizationMethod);
fprintf('Quality Detection: %s\n', string(dl_params.preprocessing.enableQualityCheck));

fprintf('Configuration loading completed.\n');

end 