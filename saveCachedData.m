function saveCachedData(allBeatInfo, training_waveforms, training_context, training_labels, testing_waveforms, testing_context, testing_labels, mat_files_info, db_allocation, dl_params)
%saveCachedData Save processed data to cache
%   This function saves the processed data and configuration information to a cache file,
%   so that it can be loaded directly next time, avoiding repeated processing.
%
%   Input:
%       allBeatInfo     - All heartbeat information
%       training_*      - Training data
%       testing_*       - Testing data
%       mat_files_info  - MAT file information
%       db_allocation   - Database allocation configuration
%       dl_params       - Deep learning parameters

fprintf('\n--- Saving Data Cache ---\n');

try
    % Create results directory (if it doesn't exist)
    cache_dir = 'results';
    if ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end
    
    % Save configuration information
    fprintf('Saving configuration cache...\n');
    processing_config = struct();
    processing_config.mat_files_info = mat_files_info;
    processing_config.db_allocation = db_allocation;
    processing_config.dl_params_relevant = struct();
    processing_config.dl_params_relevant.standardLength = dl_params.standardLength;
    
    % Use new nested parameter structure
    if isfield(dl_params, 'preprocessing') && isfield(dl_params.preprocessing, 'enableQualityCheck')
        processing_config.dl_params_relevant.enableQualityCheck = dl_params.preprocessing.enableQualityCheck;
    else
        processing_config.dl_params_relevant.enableQualityCheck = true; % Default value
    end
    
    if isfield(dl_params, 'context') && isfield(dl_params.context, 'enhanced')
        processing_config.dl_params_relevant.enhancedContext = dl_params.context.enhanced;
    else
        processing_config.dl_params_relevant.enhancedContext = true; % Default value
    end
    
    if isfield(dl_params, 'preprocessing') && isfield(dl_params.preprocessing, 'normalizationMethod')
        processing_config.dl_params_relevant.normalizationMethod = dl_params.preprocessing.normalizationMethod;
    else
        processing_config.dl_params_relevant.normalizationMethod = 'zscore'; % Default value
    end
    
    if isfield(dl_params, 'classBalance') && isfield(dl_params.classBalance, 'method')
        processing_config.dl_params_relevant.balanceMethod = dl_params.classBalance.method;
    else
        processing_config.dl_params_relevant.balanceMethod = 'downsample'; % Default value
    end
    
    if isfield(dl_params, 'classBalance') && isfield(dl_params.classBalance, 'maxRatio')
        processing_config.dl_params_relevant.maxRatio = dl_params.classBalance.maxRatio;
    else
        processing_config.dl_params_relevant.maxRatio = 2; % Default value
    end
    
    created_time = datetime('now');
    
    config_cache_file = fullfile(cache_dir, 'data_processing_config.mat');
    save(config_cache_file, 'processing_config', 'created_time');
    fprintf('✓ Configuration cache saved: %s\n', config_cache_file);
    
    % Save heartbeat information
    fprintf('Saving heartbeat information cache...\n');
    beatinfo_cache_file = fullfile(cache_dir, 'all_beat_info_cache.mat');
    save(beatinfo_cache_file, 'allBeatInfo', '-v7.3');
    fprintf('✓ Heartbeat information cache saved: %s\n', beatinfo_cache_file);
    
    % Save training data (if exists)
    if ~isempty(training_labels)
        fprintf('Saving training data cache...\n');
        training_cache_file = fullfile(cache_dir, 'training_dl_data.mat');
        save(training_cache_file, 'training_waveforms', 'training_context', 'training_labels', '-v7.3');
        fprintf('✓ Training data cache saved: %s\n', training_cache_file);
        
        % Save training data meta information
        training_meta = struct();
        training_meta.sample_count = length(training_labels);
        training_meta.waveform_length = dl_params.standardLength;
        training_meta.context_dim = size(training_context, 2);
        training_meta.labels = categories(training_labels);
        training_meta.dl_params = dl_params;
        training_meta.created_time = created_time;
        
        training_meta_file = fullfile(cache_dir, 'training_dl_meta.mat');
        save(training_meta_file, 'training_meta');
        fprintf('✓ Training data meta information saved: %s\n', training_meta_file);
    end
    
    % Save testing data (if exists)
    if ~isempty(testing_labels)
        fprintf('Saving testing data cache...\n');
        testing_cache_file = fullfile(cache_dir, 'testing_dl_data.mat');
        save(testing_cache_file, 'testing_waveforms', 'testing_context', 'testing_labels', '-v7.3');
        fprintf('✓ Testing data cache saved: %s\n', testing_cache_file);
        
        % Save testing data meta information
        testing_meta = struct();
        testing_meta.sample_count = length(testing_labels);
        testing_meta.waveform_length = dl_params.standardLength;
        testing_meta.context_dim = size(testing_context, 2);
        testing_meta.labels = categories(testing_labels);
        testing_meta.dl_params = dl_params;
        testing_meta.created_time = created_time;
        
        testing_meta_file = fullfile(cache_dir, 'testing_dl_meta.mat');
        save(testing_meta_file, 'testing_meta');
        fprintf('✓ Testing data meta information saved: %s\n', testing_meta_file);
    end
    
    fprintf('\n=== Data cache saving completed ===\n');
    fprintf('Cache file location: %s\n', cache_dir);
    fprintf('If the configuration is the same next time, the cached data will be automatically loaded\n');
    
catch ME
    fprintf('⚠ Error saving cache data: %s\n', ME.message);
    fprintf('Detailed error information:\n%s\n', getReport(ME));
    fprintf('The program will continue to run, but the cache cannot be used at the next startup\n');
end

end
