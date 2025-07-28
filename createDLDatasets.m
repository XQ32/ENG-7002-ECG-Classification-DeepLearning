function [training_waveforms, training_context, training_labels, testing_waveforms, testing_context, testing_labels] = createDLDatasets(allBeatInfo, db_allocation, dl_params)
%createDLDatasets Prepare, balance, and save data for deep learning (enhanced version)
%   This function groups allBeatInfo by database, prepares waveforms,
%   context, and labels required for deep learning for each group, then balances
%   the training data, and finally saves the data.
%   Supports configuration and data validation for multi-modal attention networks.
%
%   Input:
%       allBeatInfo     - Structure array containing all heartbeat information
%       db_allocation   - Database allocation configuration
%       dl_params       - Deep learning preprocessing parameters
%         .standardLength - Standard waveform length (required)
%         .enableQualityCheck - Whether to enable data quality check (optional, default true)
%         .enhancedContext - Whether to enable enhanced context features (optional, default true)
%         .normalizationMethod - Normalization method (optional, default 'zscore')
%         .balanceMethod - Data balancing method (optional, default 'downsample')
%         .maxRatio - Maximum class ratio (optional, default 2:1)
%         .enableValidation - Whether to enable data validation (optional, default true)
%
%   Output:
%       training_waveforms, training_context, training_labels - Training data
%       testing_waveforms, testing_context, testing_labels   - Testing data

fprintf('\n\n--- Deep Learning Migration: Step 1 - Data Preparation (Enhanced Version) ---\n');

% Parse parameters, set default values, and handle backward compatibility
if ~isfield(dl_params, 'preprocessing')
    dl_params.preprocessing = struct();
end
if ~isfield(dl_params, 'context')
    dl_params.context = struct();
end
if ~isfield(dl_params, 'classBalance')
    dl_params.classBalance = struct();
end
if ~isfield(dl_params, 'debug')
    dl_params.debug = struct();
end

% Backward compatibility: handle old parameter names
if isfield(dl_params, 'enableQualityCheck') && ~isfield(dl_params.preprocessing, 'enableQualityCheck')
    dl_params.preprocessing.enableQualityCheck = dl_params.enableQualityCheck;
end
if isfield(dl_params, 'enhancedContext') && ~isfield(dl_params.context, 'enhanced')
    dl_params.context.enhanced = dl_params.enhancedContext;
end
if isfield(dl_params, 'normalizationMethod') && ~isfield(dl_params.preprocessing, 'normalizationMethod')
    dl_params.preprocessing.normalizationMethod = dl_params.normalizationMethod;
end
if isfield(dl_params, 'balanceMethod') && ~isfield(dl_params.classBalance, 'method')
    dl_params.classBalance.method = dl_params.balanceMethod;
end
if isfield(dl_params, 'maxRatio') && ~isfield(dl_params.classBalance, 'maxRatio')
    dl_params.classBalance.maxRatio = dl_params.maxRatio;
end
if isfield(dl_params, 'enableValidation') && ~isfield(dl_params.debug, 'enableValidation')
    dl_params.debug.enableValidation = dl_params.enableValidation;
end

% Set default values
if ~isfield(dl_params.preprocessing, 'enableQualityCheck'), dl_params.preprocessing.enableQualityCheck = true; end
if ~isfield(dl_params.context, 'enhanced'), dl_params.context.enhanced = true; end
if ~isfield(dl_params.preprocessing, 'normalizationMethod'), dl_params.preprocessing.normalizationMethod = 'zscore'; end
if ~isfield(dl_params.classBalance, 'method'), dl_params.classBalance.method = 'downsample'; end
if ~isfield(dl_params.classBalance, 'maxRatio'), dl_params.classBalance.maxRatio = 2; end
if ~isfield(dl_params.debug, 'enableValidation'), dl_params.debug.enableValidation = true; end

fprintf('Configuration parameters: Standard length=%d, Quality check=%s, Enhanced context=%s, Normalization=%s\n', ...
    dl_params.standardLength, string(dl_params.preprocessing.enableQualityCheck), ...
    string(dl_params.context.enhanced), dl_params.preprocessing.normalizationMethod);

% Initialize output variables
training_waveforms = {};
training_context = [];
training_labels = categorical([]);
testing_waveforms = {};
testing_context = [];
testing_labels = categorical([]);

% Data validation: check input integrity
if dl_params.debug.enableValidation
    fprintf('\n--- Input Data Validation ---\n');
    
    actual_beat_count = length(allBeatInfo);
    fprintf('Total number of heartbeats: %d\n', actual_beat_count);
    
    if actual_beat_count == 0
        warning('allBeatInfo is empty, cannot create dataset');
        return;
    end
    
    % Validate necessary fields
    required_fields = {'segment', 'beatType', 'originalDatabaseName', 'fs', 'segmentStartIndex', 'rIndex'};
    missing_fields = cell(1, length(required_fields)); % Pre-allocate
    missing_count = 0;
    for i = 1:length(required_fields)
        if ~isfield(allBeatInfo, required_fields{i})
            missing_count = missing_count + 1;
            missing_fields{missing_count} = required_fields{i};
        end
    end
    missing_fields = missing_fields(1:missing_count); % Trim to valid part
    
    if ~isempty(missing_fields)
        error('allBeatInfo is missing necessary fields: %s', strjoin(missing_fields, ', '));
    end
    
    % Validate database allocation configuration
    if ~isfield(db_allocation, 'training') || ~isfield(db_allocation, 'testing')
        error('db_allocation must contain training and testing fields');
    end
    
    % Statistics of database distribution
    unique_dbs = unique({allBeatInfo.originalDatabaseName});
    fprintf('Database distribution: %s\n', strjoin(unique_dbs, ', '));
    fprintf('Training databases: %s\n', strjoin(db_allocation.training, ', '));
    fprintf('Testing databases: %s\n', strjoin(db_allocation.testing, ', '));
    
    % Check label distribution
    all_labels = {allBeatInfo.beatType};
    unique_labels = unique(all_labels);
    fprintf('Label distribution: ');
    for i = 1:length(unique_labels)
        count = sum(strcmp(all_labels, unique_labels{i}));
        fprintf('%s=%d (%.1f%%) ', unique_labels{i}, count, count/actual_beat_count*100);
    end
    fprintf('\n');
    
    fprintf('Input data validation passed\n');
end

actual_beat_count = length(allBeatInfo);
if actual_beat_count > 0
    fprintf('\n--- Processing by database and sampling frequency groups ---\n');
    
    % Efficient grouping: extract database name and sampling frequency
    db_names = cell(actual_beat_count, 1);
    fs_values = zeros(actual_beat_count, 1);
    for i = 1:actual_beat_count
        db_names{i} = allBeatInfo(i).originalDatabaseName;
        fs_values(i) = allBeatInfo(i).fs;
    end
    [unique_combinations, ~, group_indices_map] = unique(strcat(db_names, '_', string(fs_values)));
    
    uniqueDBFSGroups = table();
    uniqueDBFSGroups.originalDatabaseName = cell(length(unique_combinations), 1);
    uniqueDBFSGroups.fs = zeros(length(unique_combinations), 1);
    for i = 1:length(unique_combinations)
        first_occurrence = find(group_indices_map == i, 1);
        uniqueDBFSGroups.originalDatabaseName{i} = db_names{first_occurrence};
        uniqueDBFSGroups.fs(i) = fs_values(first_occurrence);
    end
    
    fprintf('Found %d database-sampling frequency combinations\n', height(uniqueDBFSGroups));
    
    % Memory optimization: use temporary variables to store grouped data
    all_training_data = struct('waveforms', {{}}, 'context', {{}}, 'labels', {{}});
    all_testing_data = struct('waveforms', {{}}, 'context', {{}}, 'labels', {{}});
    
    % Process each database group
    for group_idx = 1:height(uniqueDBFSGroups)
        current_group_db_name = uniqueDBFSGroups.originalDatabaseName{group_idx};
        current_group_fs = uniqueDBFSGroups.fs(group_idx);
        
        fprintf('  Processing group %d/%d: Database=''%s'', fs=%d Hz\n', ...
            group_idx, height(uniqueDBFSGroups), current_group_db_name, current_group_fs);
        
        % Use logical indexing for efficiency
        group_indices = strcmp(db_names, current_group_db_name) & (fs_values == current_group_fs);
        beats_for_current_group = allBeatInfo(group_indices);
        
        if isempty(beats_for_current_group)
            fprintf('    No heartbeat data in this group, skipping.\n');
            continue;
        end
        
        % Call enhanced data preprocessing function
        try
            [waveforms, context, labels] = processBeatsForDL(beats_for_current_group, current_group_fs, dl_params);
        catch ME
            fprintf('    Error: Failed to process group %s: %s\n', current_group_db_name, ME.message);
            continue;
        end
        
        % Data integrity validation
        if dl_params.debug.enableValidation
            if length(waveforms) ~= size(context, 1) || length(waveforms) ~= length(labels)
                warning('Data dimension mismatch in group %s, skipping', current_group_db_name);
                continue;
            end
            
            % Check for valid data
            if isempty(labels)
                fprintf('    No valid data after processing this group, skipping.\n');
                continue;
            end
        end
        
        % Remove heartbeats at group boundaries (NaN context features)
        if ~isempty(context)
            valid_rows = ~any(isnan(context), 2);
            num_removed = sum(~valid_rows);
            if num_removed > 0
                waveforms = waveforms(valid_rows);
                context = context(valid_rows, :);
                labels = labels(valid_rows);
                fprintf('    Removed %d boundary heartbeats, retaining %d valid heartbeats\n', num_removed, length(labels));
            else
                fprintf('    All %d heartbeats are valid\n', length(labels));
            end
        end
        
        if isempty(labels)
            continue;
        end

        % Add to corresponding set based on database allocation
        if ismember(current_group_db_name, db_allocation.training)
            all_training_data.waveforms{end+1} = waveforms;
            all_training_data.context{end+1} = context;
            all_training_data.labels{end+1} = labels;
            fprintf('    → Training set: +%d samples\n', length(labels));
        elseif ismember(current_group_db_name, db_allocation.testing)
            all_testing_data.waveforms{end+1} = waveforms;
            all_testing_data.context{end+1} = context;
            all_testing_data.labels{end+1} = labels;
            fprintf('    → Testing set: +%d samples\n', length(labels));
        else
            fprintf('    Warning: Database %s not allocated to training or testing set\n', current_group_db_name);
        end
    end
    
    % Merge all training and testing data
    fprintf('\n--- Data Set Merging ---\n');
    if ~isempty(all_training_data.waveforms)
        training_waveforms = vertcat(all_training_data.waveforms{:});
        training_context = vertcat(all_training_data.context{:});
        training_labels = vertcat(all_training_data.labels{:});
        fprintf('Training set merging complete: %d samples\n', length(training_labels));
    else
        fprintf('Training set is empty\n');
    end
    
    if ~isempty(all_testing_data.waveforms)
        testing_waveforms = vertcat(all_testing_data.waveforms{:});
        testing_context = vertcat(all_testing_data.context{:});
        testing_labels = vertcat(all_testing_data.labels{:});
        fprintf('Testing set merging complete: %d samples\n', length(testing_labels));
    else
        fprintf('Testing set is empty\n');
    end
    
    % Memory cleanup
    clear all_training_data all_testing_data db_names fs_values unique_combinations group_indices_map;
    
    fprintf('\nDeep learning data merging complete.\n');
    fprintf('  - Training set: %d samples\n', length(training_labels));
    fprintf('  - Testing set: %d samples\n', length(testing_labels));
else
    fprintf('\nallBeatInfo is empty, skipping deep learning data preparation.\n');
end

% Training data balancing
fprintf('\n\n--- Training Data Balancing ---\n');
if ~isempty(training_labels)
    unique_train_labels = categories(training_labels);
    fprintf('Balancing method: %s, Max ratio: %.1f:1\n', dl_params.classBalance.method, dl_params.classBalance.maxRatio);
    
    % Count each class
    label_counts = zeros(length(unique_train_labels), 1);
    label_indices = cell(length(unique_train_labels), 1);
    
    for i = 1:length(unique_train_labels)
        label_indices{i} = find(training_labels == unique_train_labels{i});
        label_counts(i) = length(label_indices{i});
        fprintf('  %s: %d samples\n', unique_train_labels{i}, label_counts(i));
    end
    
    % Perform data balancing
    switch dl_params.classBalance.method
        case 'downsample'
            % Downsampling balancing
            [min_count, ~] = min(label_counts); % Use ~ to ignore unused min_idx
            max_allowed_count = min_count * dl_params.classBalance.maxRatio;

            % Pre-allocate selected_indices array
            total_estimated_samples = sum(min(label_counts, max_allowed_count));
            selected_indices = zeros(total_estimated_samples, 1);
            current_idx = 1;
            rng(42); % Set random seed for reproducibility

            for i = 1:length(unique_train_labels)
                current_count = label_counts(i);
                current_indices = label_indices{i};

                if current_count > max_allowed_count
                    % Randomly select samples for downsampling
                    selected_count = round(max_allowed_count);
                    perm_indices = randperm(current_count, selected_count);
                    selected_samples = current_indices(perm_indices);
                    selected_indices(current_idx:current_idx+selected_count-1) = selected_samples;
                    current_idx = current_idx + selected_count;
                    fprintf('  %s: Downsampled from %d to %d\n', unique_train_labels{i}, current_count, selected_count);
                else
                    selected_indices(current_idx:current_idx+current_count-1) = current_indices;
                    current_idx = current_idx + current_count;
                    fprintf('  %s: Kept %d samples\n', unique_train_labels{i}, current_count);
                end
            end

            % Trim to valid part
            selected_indices = selected_indices(1:current_idx-1);
            
            % Apply balancing selection
            selected_indices = sort(selected_indices);
            training_waveforms = training_waveforms(selected_indices);
            training_context = training_context(selected_indices, :);
            training_labels = training_labels(selected_indices);
            
        case 'none'
            fprintf('No data balancing performed\n');
            
        case 'weights'
            fprintf('Using class weights method for data resampling\n');
            
            % Calculate weight ratios and apply resampling
            weights = dl_params.classBalance.classWeights;
            fprintf('Weight configuration: [%.1f, %.1f]\n', weights(1), weights(2));
            
            % Find the smallest class
            [min_count, min_idx] = min(label_counts);
            
            % Pre-allocate selected_indices array
            total_estimated_samples = 0;
            target_counts = zeros(length(unique_train_labels), 1);
            
            % Calculate target count for each class
            for i = 1:length(unique_train_labels)
                if i == min_idx
                    % Smallest class: target count determined by its weight
                    target_counts(i) = min_count;
                else
                    % Other classes: target count determined by weight ratio
                    weight_ratio = weights(i) / weights(min_idx);
                    target_counts(i) = min(label_counts(i), round(min_count / weight_ratio));
                end
                total_estimated_samples = total_estimated_samples + target_counts(i);
                fprintf('  %s: Target count %d (original count %d)\n', unique_train_labels{i}, target_counts(i), label_counts(i));
            end
            
            selected_indices = zeros(total_estimated_samples, 1);
            current_idx = 1;
            rng(42); % Set random seed
            
            % Perform resampling
            for i = 1:length(unique_train_labels)
                current_count = label_counts(i);
                current_indices = label_indices{i};
                target_count = target_counts(i);
                
                if current_count > target_count
                    % Downsample
                    perm_indices = randperm(current_count, target_count);
                    selected_samples = current_indices(perm_indices);
                    fprintf('  %s: Downsampled from %d to %d\n', unique_train_labels{i}, current_count, target_count);
                else
                    % Keep as is or upsample
                    if target_count > current_count
                        % Upsample (repeat sampling)
                        repeat_factor = ceil(target_count / current_count);
                        repeated_indices = repmat(current_indices, repeat_factor, 1);
                        selected_samples = repeated_indices(1:target_count);
                        fprintf('  %s: Upsampled from %d to %d\n', unique_train_labels{i}, current_count, target_count);
                    else
                        selected_samples = current_indices;
                        fprintf('  %s: Kept %d samples\n', unique_train_labels{i}, target_count);
                    end
                end
                
                selected_indices(current_idx:current_idx+target_count-1) = selected_samples;
                current_idx = current_idx + target_count;
            end
            
            % Apply balancing selection
            selected_indices = sort(selected_indices);
            training_waveforms = training_waveforms(selected_indices);
            training_context = training_context(selected_indices, :);
            training_labels = training_labels(selected_indices);
            
        otherwise
            warning('Unknown balancing method: %s, skipping balancing', dl_params.classBalance.method);
    end
    
    % Display statistics after balancing
    fprintf('\nData distribution after balancing:\n');
    balanced_total = length(training_labels);
    for i = 1:length(unique_train_labels)
        balanced_count = sum(training_labels == unique_train_labels{i});
        fprintf('  %s: %d samples (%.1f%%)\n', unique_train_labels{i}, ...
            balanced_count, balanced_count/balanced_total*100);
    end
    fprintf('  Total: %d samples\n', balanced_total);
else
    fprintf('Training set is empty, cannot perform data balancing.\n');
end

% Final data validation
if dl_params.debug.enableValidation && ~isempty(training_labels)
    fprintf('\n--- Final Data Validation ---\n');
    
    % Check training set
    if length(training_waveforms) ~= size(training_context, 1) || ...
       length(training_waveforms) ~= length(training_labels)
        error('Training set data dimension mismatch');
    end
    
    % Check testing set
    if ~isempty(testing_labels)
        if length(testing_waveforms) ~= size(testing_context, 1) || ...
           length(testing_waveforms) ~= length(testing_labels)
            error('Testing set data dimension mismatch');
        end
    end
    
    % Check context feature dimension consistency
    if ~isempty(testing_context) && size(training_context, 2) ~= size(testing_context, 2)
        error('Context feature dimensions of training and testing sets do not match');
    end
    
    fprintf('Final validation passed: Training set %d samples, Testing set %d samples\n', ...
        length(training_labels), length(testing_labels));
    
    if dl_params.context.enhanced
        fprintf('Context feature dimension: %d (Enhanced mode)\n', size(training_context, 2));
    else
        fprintf('Context feature dimension: %d (Basic mode)\n', size(training_context, 2));
    end
end

% Save data to file
fprintf('\n--- Saving Data to File ---\n');
try
    if ~isempty(training_labels)
        % Create results directory (if it doesn't exist)
        if ~exist('results', 'dir')
            mkdir('results');
        end
        
        save('results/training_dl_data.mat', 'training_waveforms', 'training_context', 'training_labels', '-v7.3');
        fprintf('✓ Training data saved: results/training_dl_data.mat\n');
        
        % Save training data metadata
        training_meta = struct();
        training_meta.sample_count = length(training_labels);
        training_meta.waveform_length = dl_params.standardLength;
        training_meta.context_dim = size(training_context, 2);
        training_meta.labels = categories(training_labels);
        training_meta.dl_params = dl_params;
        training_meta.created_time = datetime('now');
        
        save('results/training_dl_meta.mat', 'training_meta');
        fprintf('✓ Training data metadata saved: results/training_dl_meta.mat\n');
    else
        fprintf('⚠ Training data is empty, skipping save.\n');
    end
    
    if ~isempty(testing_labels)
        save('results/testing_dl_data.mat', 'testing_waveforms', 'testing_context', 'testing_labels', '-v7.3');
        fprintf('✓ Testing data saved: results/testing_dl_data.mat\n');
        
        % Save testing data metadata
        testing_meta = struct();
        testing_meta.sample_count = length(testing_labels);
        testing_meta.waveform_length = dl_params.standardLength;
        testing_meta.context_dim = size(testing_context, 2);
        testing_meta.labels = categories(testing_labels);
        testing_meta.dl_params = dl_params;
        testing_meta.created_time = datetime('now');
        
        save('results/testing_dl_meta.mat', 'testing_meta');
        fprintf('✓ Testing data metadata saved: results/testing_dl_meta.mat\n');
    else
        fprintf('⚠ Testing data is empty, skipping save.\n');
    end
catch ME_save_dl
    fprintf('⚠ Error saving data: %s\n', ME_save_dl.message);
    fprintf('Detailed error information:\n%s\n', getReport(ME_save_dl));
end

fprintf('\n=== Dataset Creation Complete ===\n');

end