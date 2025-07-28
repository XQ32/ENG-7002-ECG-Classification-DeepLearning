function [useCache, allBeatInfo, training_waveforms, training_context, training_labels, testing_waveforms, testing_context, testing_labels] = checkAndLoadCachedData(mat_files_info, db_allocation, dl_params)
%checkAndLoadCachedData Check and load cached data
%   This function checks if previously processed data cache exists, and loads directly if dataset configuration is the same,
%   avoiding repeated execution of data processing steps 2-4.
%
%   Input:
%       mat_files_info  - MAT file information
%       db_allocation   - Database allocation configuration
%       dl_params       - Deep learning parameters
%
%   Output:
%       useCache        - Whether to use cached data
%       allBeatInfo     - All heartbeat information (if using cache)
%       training_*      - Training data (if using cache)
%       testing_*       - Testing data (if using cache)

fprintf('\n--- Check Data Cache ---\n');

% Initialize output variables
useCache = false;
allBeatInfo = [];
training_waveforms = {};
training_context = [];
training_labels = categorical([]);
testing_waveforms = {};
testing_context = [];
testing_labels = categorical([]);

% Define cache file paths
cache_dir = 'results';
config_cache_file = fullfile(cache_dir, 'data_processing_config.mat');
beatinfo_cache_file = fullfile(cache_dir, 'all_beat_info_cache.mat');
training_cache_file = fullfile(cache_dir, 'training_dl_data.mat');
testing_cache_file = fullfile(cache_dir, 'testing_dl_data.mat');

% Check if cache files exist
if ~exist(config_cache_file, 'file') || ~exist(beatinfo_cache_file, 'file') || ...
   ~exist(training_cache_file, 'file') || ~exist(testing_cache_file, 'file')
    fprintf('Cache files incomplete, need to reprocess data\n');
    return;
end

try
    % Load configuration cache
    fprintf('Checking configuration cache...\n');
    cached_config = load(config_cache_file);

    % Create hash value for current configuration
    current_config = struct();
    current_config.mat_files_info = mat_files_info;
    current_config.db_allocation = db_allocation;
    current_config.dl_params_relevant = struct();
    current_config.dl_params_relevant.standardLength = dl_params.standardLength;
    current_config.dl_params_relevant.enableQualityCheck = dl_params.preprocessing.enableQualityCheck;
    current_config.dl_params_relevant.enhancedContext = dl_params.context.enhanced;
    current_config.dl_params_relevant.normalizationMethod = dl_params.preprocessing.normalizationMethod;
    current_config.dl_params_relevant.balanceMethod = dl_params.classBalance.method;
    current_config.dl_params_relevant.maxRatio = dl_params.classBalance.maxRatio;
    
    % Compare if configurations are the same
    if isequal(cached_config.processing_config, current_config)
        fprintf('✓ Configuration matches, loading cached data...\n');
        
        % Load all cached data
        fprintf('  Loading heartbeat information cache...\n');
        beatinfo_data = load(beatinfo_cache_file);
        allBeatInfo = beatinfo_data.allBeatInfo;
        
        fprintf('  Loading training data cache...\n');
        training_data = load(training_cache_file);
        training_waveforms = training_data.training_waveforms;
        training_context = training_data.training_context;
        training_labels = training_data.training_labels;
        
        fprintf('  Loading testing data cache...\n');
        testing_data = load(testing_cache_file);
        testing_waveforms = testing_data.testing_waveforms;
        testing_context = testing_data.testing_context;
        testing_labels = testing_data.testing_labels;
        
        % Display cached data information
        fprintf('\n=== Cached Data Information ===\n');
        fprintf('Cache creation time: %s\n', char(cached_config.created_time));
        fprintf('Total number of heartbeats: %d\n', length(allBeatInfo));
        fprintf('Training set: %d samples\n', length(training_labels));
        fprintf('Testing set: %d samples\n', length(testing_labels));
        
        % Display label distribution
        if ~isempty(training_labels)
            fprintf('\nTraining set label distribution:\n');
            unique_labels = categories(training_labels);
            for i = 1:length(unique_labels)
                count = sum(training_labels == unique_labels{i});
                fprintf('  %s: %d samples (%.1f%%)\n', unique_labels{i}, ...
                    count, count/length(training_labels)*100);
            end
        end
        
        if ~isempty(testing_labels)
            fprintf('\nTesting set label distribution:\n');
            unique_labels = categories(testing_labels);
            for i = 1:length(unique_labels)
                count = sum(testing_labels == unique_labels{i});
                fprintf('  %s: %d samples (%.1f%%)\n', unique_labels{i}, ...
                    count, count/length(testing_labels)*100);
            end
        end
        
        % Display context feature information
        if ~isempty(training_context)
            fprintf('\nContext feature information:\n');
            fprintf('  Feature dimension: %d\n', size(training_context, 2));
            if dl_params.context.enhanced && size(training_context, 2) == 4
                fprintf('  Feature type: Enhanced mode (RR_Prev, RR_Post, HR_Variability, Rhythm_Stability)\n');
            elseif size(training_context, 2) == 2
                fprintf('  Feature type: Basic mode (RR_Prev, RR_Post)\n');
            else
                fprintf('  Feature type: Custom (%d-dim)\n', size(training_context, 2));
            end
        end
        
        fprintf('\n✓ Cached data loading complete, skipping steps 2-4, proceeding directly to model training\n');
        useCache = true;
        
    else
        fprintf('Configuration mismatch, data needs to be reprocessed\n');
        fprintf('Main differences may include:\n');
        fprintf('  - Database file list changed\n');
        fprintf('  - Database allocation strategy changed\n');
        fprintf('  - Data preprocessing parameters changed\n');
    end
    
catch ME
    fprintf('Error loading cached data: %s\n', ME.message);
    fprintf('Reprocessing data\n');
    useCache = false;
end

end
