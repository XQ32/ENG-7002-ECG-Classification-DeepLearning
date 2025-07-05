%% Main Function: Load ECG data from .mat files, detect heartbeats, classify, and extract features for machine learning
% This program uses ecgFilter for filtering
% and then uses detectAndClassifyHeartbeats to detect and classify heartbeats, and extract features for machine learning
%
% Memory Optimized Version:
% - Uses dynamic memory allocation to avoid pre-allocating oversized arrays
% - Timely cleanup of temporary variables
% - Uses in-place compression to reduce memory copying
% - Optimizes loops and data structure access

clc;
clear;
close all;



%% Parameter Settings

% Database allocation configuration - easily modifiable parameters to specify which databases are for training/testing
db_allocation = struct();

% Databases for training
db_allocation.training = {'MIT_data','mit_bih_supraventricular_arrhythmia_database_1_0_0_data', 'mit_bih_long_term_ecg_database_1_0_0_data', 'european_st_t_database_1_0_0_data', 'leipzig_heart_center_ecg_database_arrhythmias_in_children_and_p', ...
    'cu_ventricular_tachyarrhythmia_database_1_0_0_data', ...
};       
     
% Databases for testing
db_allocation.testing = {'INCART_data'}; 

% List of MAT database files (assumed to be in the 'results' folder)
mat_files_info = {
    struct('fileName', 'MIT_data.mat', 'structName', 'MIT_data', 'powerLineFreq', 60), ...
    struct('fileName', 'INCART_data.mat', 'structName', 'INCART_data', 'powerLineFreq', 50), ...
    struct('fileName', 'leipzig-heart-center-ecg-database-arrhythmias-in-children-and-patients-with-congenital-heart-disease-1.0.0_data.mat', 'structName', 'leipzig_heart_center_ecg_database_arrhythmias_in_children_and_p', 'powerLineFreq', 50), ...
    struct('fileName', 'mit-bih-supraventricular-arrhythmia-database-1.0.0_data.mat', 'structName', 'mit_bih_supraventricular_arrhythmia_database_1_0_0_data', 'powerLineFreq', 60), ...
    struct('fileName', 'mit-bih-long-term-ecg-database-1.0.0_data.mat', 'structName', 'mit_bih_long_term_ecg_database_1_0_0_data', 'powerLineFreq', 60), ...
    struct('fileName', 'cu-ventricular-tachyarrhythmia-database-1.0.0_data.mat', 'structName', 'cu_ventricular_tachyarrhythmia_database_1_0_0_data', 'powerLineFreq', 60), ...
    struct('fileName', 'european-st-t-database-1.0.0_data.mat', 'structName', 'european_st_t_database_1_0_0_data', 'powerLineFreq', 50), ...
    
    % struct('fileName', 'long-term-af-database-1.0.0_data.mat', 'structName', 'long_term_af_database_1_0_0_data', 'powerLineFreq', 60), ...
    % struct('fileName', 'paroxysmal-atrial-fibrillation-events-detection_data.mat', 'structName', 'paroxysmal_atrial_fibrillation_events_detection_data', 'powerLineFreq', 50), ...
    % struct('fileName', 'mit-bih-malignant-ventricular-ectopy-database-1.0.0_data.mat', 'structName', 'mit_bih_malignant_ventricular_ectopy_database_1_0_0_data', 'powerLineFreq', 60), ...
    % struct('fileName', 'sudden-cardiac-death-holter-database-1.0.0_data.mat', 'structName', 'sudden_cardiac_death_holter_database_1_0_0_data', 'powerLineFreq', 60), ...
    % struct('fileName', 'shdb-af-a-japanese-holter-ecg-database-of-atrial-fibrillation-1.0.1_data.mat', 'structName', 'shdb_af_a_japanese_holter_ecg_database_of_atrial_fibrillation_1_0_1_data', 'powerLineFreq', 50), ...
    % struct('fileName', 'recordings-excluded-from-the-nsr-db-1.0.0_data.mat', 'structName', 'recordings_excluded_from_the_nsr_db_1_0_0_data', 'powerLineFreq', 60), ...

};
results_dir = 'results'; % Directory where .mat files are located

%% Initialize Variables - Memory Optimized Version
% Use a dynamic growth strategy to avoid pre-allocating oversized memory
initial_struct = struct('beatType', 'Other', 'segment', [], 'rIndex', 0, 'pIndex', NaN, ...
                'qIndex', NaN, 'sIndex', NaN, 'tIndex', NaN, 'pEnd', NaN, 'tEnd', NaN, ...
                'segmentStartIndex', 0, 'fs', 0, 'originalRecordName', '', ...
                'originalDatabaseName', '', 'originalAnnotationChar', '');

% Dynamic memory management parameters
chunk_size = 100000;  % Increase by 100,000 rows each time
allBeatInfo = repmat(initial_struct, chunk_size, 1);  % Initially allocate 100,000 rows
actual_beat_count = 0;  % Track the actual number of heartbeats added
allocated_size = chunk_size;  % Track the currently allocated size

fprintf('Using dynamic memory allocation strategy, initially allocating %d rows, will expand as needed\n', chunk_size);

%% Process Data Records One by One
fprintf('Starting to process database .mat files, detect heartbeats, and extract features...\n');

for db_idx = 1:length(mat_files_info)
    current_db_info = mat_files_info{db_idx};
    mat_file_path = fullfile(results_dir, current_db_info.fileName);
    
    fprintf('\nLoading database file: %s\n', current_db_info.fileName);
    
    if ~exist(mat_file_path, 'file')
        fprintf('Warning: File %s not found in ''%s'' directory. Skipping this database.\n', current_db_info.fileName, results_dir);
        continue;
    end
    
    try
        loaded_data = load(mat_file_path);
    catch ME_load
        fprintf('Warning: Error loading file %s: %s. Skipping this database.\n', current_db_info.fileName, ME_load.message);
        continue;
    end
    
    if ~isfield(loaded_data, current_db_info.structName)
        fprintf('Warning: Expected struct variable ''%s'' not found in %s. Skipping this database.\n', current_db_info.fileName, current_db_info.structName);
        % Clean up memory: release loaded but unused data
        clear loaded_data;
        continue;
    end
    
    db_records_array = loaded_data.(current_db_info.structName);
    fprintf('Database %s contains %d records.\n', current_db_info.structName, length(db_records_array));
    
    % Clean up memory: release data in loaded_data other than the target struct
    clear loaded_data;

    for rec_idx = 1:length(db_records_array)
        current_record = db_records_array(rec_idx);
        recordName = current_record.name;
        fs = current_record.fs;
        
        fprintf('  Processing record: %s (from %s, fs: %d Hz)\n', recordName, current_db_info.fileName, fs);
        
        if isempty(current_record.signal_data)
            fprintf('    Warning: Signal data for record %s is empty. Skipping this record.\n', recordName);
            continue;
        end
        M = current_record.signal_data(:, 1); % Default to the first lead
        
        if isempty(current_record.annotations) || isempty(current_record.anntype)
             fprintf('    Warning: Annotations or anntype for record %s are empty. Skipping this record.\n', recordName);
            continue;
        end
        
        ATRTIMED = double(current_record.annotations) / fs;
        ANNOT_CHAR_orig = current_record.anntype; % Character-type annotations

        % Convert character-type ANNOT_CHAR_orig to string-type ANNOTD
        % New binary classification: PVC and Other (PAC is merged into Other)
        ANNOTD = cell(size(ANNOT_CHAR_orig, 1), 1);
        for k_ann = 1:size(ANNOT_CHAR_orig, 1)
            ann_char_single = strtrim(ANNOT_CHAR_orig(k_ann,:)); 
            switch upper(ann_char_single)
                case 'V'
                    ANNOTD{k_ann} = 'PVC'; % PVC
                otherwise
                    ANNOTD{k_ann} = 'Other'; % Other heartbeats (including original Normal, PAC, other abnormal heartbeats, etc.)
            end
        end
        
        % Use try-catch to ensure the program continues even if detectAndClassifyHeartbeats fails
        try
            % Call ecgFilter according to the database's power line frequency parameter
            power_line_freq = current_db_info.powerLineFreq;
            fprintf('    Using filter to remove %d Hz power line interference\n', power_line_freq);
            [ecg_filtered, filter_info] = ecgFilter(M, fs, 2, power_line_freq);
            
            % Optional: if filter information needs to be output to the console
            if isfield(filter_info, 'has_2nd_harmonic_filter') && filter_info.has_2nd_harmonic_filter
                fprintf('    Applied %d Hz (fundamental) and %d Hz (2nd harmonic) notch filters\n', ...
                    power_line_freq, power_line_freq*2);
            else
                fprintf('    Applied %d Hz notch filter\n', power_line_freq);
            end

            [heartbeatSegments_rec, beatInfo_rec_numeric] = detectAndClassifyHeartbeats(ecg_filtered, ATRTIMED, ANNOTD, fs);

            % Check if the returned result is empty
            if isempty(beatInfo_rec_numeric)
                fprintf('  Warning: Heartbeat detection result for record %s is empty, skipping this record.\n', recordName);
                % Clean up memory: release temporary variables for the current record
                clear M ecg_filtered heartbeatSegments_rec ATRTIMED ANNOT_CHAR_orig ANNOTD;
                continue;
            end

            % Add original character annotations and source information to beatInfo_rec_numeric
            if ~isempty(beatInfo_rec_numeric)
                tempBeatInfoArray = repmat(struct(...
                    'beatType', 'Other', 'segment', [], 'rIndex', 0, 'pIndex', NaN, ...
                    'qIndex', NaN, 'sIndex', NaN, 'tIndex', NaN, 'pEnd', NaN, 'tEnd', NaN, ...
                    'segmentStartIndex', 0, 'fs', fs, ...
                    'originalRecordName', recordName, 'originalDatabaseName', current_db_info.structName, ...
                    'originalAnnotationChar', ''), ...
                    length(beatInfo_rec_numeric), 1);

                for j = 1:length(beatInfo_rec_numeric)
                    tempBeatInfoArray(j).beatType = beatInfo_rec_numeric(j).beatType;
                    tempBeatInfoArray(j).segment = beatInfo_rec_numeric(j).segment;
                    tempBeatInfoArray(j).rIndex = beatInfo_rec_numeric(j).rIndex;
                    tempBeatInfoArray(j).pIndex = beatInfo_rec_numeric(j).pIndex;
                    tempBeatInfoArray(j).qIndex = beatInfo_rec_numeric(j).qIndex;
                    tempBeatInfoArray(j).sIndex = beatInfo_rec_numeric(j).sIndex;
                    tempBeatInfoArray(j).tIndex = beatInfo_rec_numeric(j).tIndex;
                    tempBeatInfoArray(j).pEnd = beatInfo_rec_numeric(j).pEnd;
                    tempBeatInfoArray(j).tEnd = beatInfo_rec_numeric(j).tEnd;
                    tempBeatInfoArray(j).segmentStartIndex = beatInfo_rec_numeric(j).segmentStartIndex;
                    tempBeatInfoArray(j).fs = fs; % Store the fs of the current record
                    tempBeatInfoArray(j).originalRecordName = recordName;
                    tempBeatInfoArray(j).originalDatabaseName = current_db_info.structName;

                    % Find original character annotation
                    r_peak_in_original_ecg = beatInfo_rec_numeric(j).segmentStartIndex + beatInfo_rec_numeric(j).rIndex - 1;
                    [~, closest_idx_in_orig_ann] = min(abs(double(current_record.annotations) - r_peak_in_original_ecg));
                    
                    % Check for exact match or very close (threshold can be adjusted, e.g., a few sample points)
                    if abs(double(current_record.annotations(closest_idx_in_orig_ann)) - r_peak_in_original_ecg) <= 2 % Allow +/-2 samples of error
                        tempBeatInfoArray(j).originalAnnotationChar = strtrim(ANNOT_CHAR_orig(closest_idx_in_orig_ann,:));
                    else
                        tempBeatInfoArray(j).originalAnnotationChar = '?'; % No exact match found
                        fprintf('    Warning: Record %s, R-peak@%d could not be precisely matched back to the original character annotation. Closest distance %d sample points.\n', ...
                                recordName, r_peak_in_original_ecg, ...
                                abs(double(current_record.annotations(closest_idx_in_orig_ann)) - r_peak_in_original_ecg));
                    end
                end
                beatInfo_rec = tempBeatInfoArray;
                
                % Clean up memory: release temporary arrays
                clear tempBeatInfoArray beatInfo_rec_numeric;
            else
                beatInfo_rec = [];
            end

            % Dynamic memory expansion - check if allBeatInfo needs to be expanded
            new_beats_count = length(beatInfo_rec);
            if actual_beat_count + new_beats_count > allocated_size
                % Memory expansion needed
                additional_chunks = ceil((actual_beat_count + new_beats_count - allocated_size) / chunk_size);
                new_allocation = allocated_size + additional_chunks * chunk_size;
                fprintf('    Expanding memory: increasing from %d rows to %d rows (+%d rows)\n', allocated_size, new_allocation, additional_chunks * chunk_size);
                
                % Create a new larger array
                temp_allBeatInfo = repmat(initial_struct, new_allocation, 1);
                % Copy existing data
                temp_allBeatInfo(1:actual_beat_count) = allBeatInfo(1:actual_beat_count);
                % Free memory of the old array
                clear allBeatInfo;
                % Use the new array
                allBeatInfo = temp_allBeatInfo;
                allocated_size = new_allocation;
                % Clean up temporary variables
                clear temp_allBeatInfo;
            end
            
            % Add new heartbeat information
            allBeatInfo(actual_beat_count + 1:actual_beat_count + new_beats_count) = beatInfo_rec;
            actual_beat_count = actual_beat_count + new_beats_count;
    
            % if ~isempty(beatInfo_rec)
            %     types_char = arrayfun(@(x) x.originalAnnotationChar, beatInfo_rec, 'UniformOutput', false);
            %     numNormal_char = sum(ismember(types_char, {'N', 'L', 'R', '/'}));
            %     numPVC_char    = sum(strcmp(types_char, 'V'));
            %     numPAC_char    = sum(ismember(types_char, {'A', 'S'}));
            %     numUnknown_char= sum(strcmp(types_char, '?'));
            % 
            %     fprintf('  Record %s processing complete. Extracted heartbeats (character stats): Normal-like=%d, PVC=%d, PAC=%d, Unmatched char=%d\n', ...
            %         recordName, numNormal_char, numPVC_char, numPAC_char, numUnknown_char);
            %     fprintf('**************************************************************************\n');
            % else
            %     fprintf('  No heartbeats extracted from record %s.\n', recordName);
            % end
            
            % Clean up memory: release all temporary variables after processing the current record
            clear M ecg_filtered heartbeatSegments_rec beatInfo_rec ATRTIMED ANNOT_CHAR_orig ANNOTD types_char;
            
        catch ME_processing
            fprintf('  An error occurred while processing record %s: %s\n  Skipping this record and continuing.\n', recordName, ME_processing.message);
            % Clean up memory: also clean up temporary variables in case of error
            clear M ecg_filtered heartbeatSegments_rec ATRTIMED ANNOT_CHAR_orig ANNOTD;
            continue;
        end
    end
    
    % Database processing finished, clean up temporary variables
    clear db_records_array current_db_info;
end

%% Data processing finished, optimize memory usage
fprintf('\n=== Data Processing Complete, Starting Memory Optimization ===\n');
fprintf('Actual usage: %d rows, Allocated: %d rows, Memory utilization: %.1f%%\n', ...
    actual_beat_count, allocated_size, (actual_beat_count/allocated_size)*100);

% If memory utilization is below 50%, shrink the memory
if actual_beat_count < allocated_size * 0.5
    fprintf('Memory utilization is low, shrinking array to save memory...\n');
    temp_allBeatInfo = repmat(initial_struct, actual_beat_count, 1);
    temp_allBeatInfo(1:actual_beat_count) = allBeatInfo(1:actual_beat_count);
    clear allBeatInfo;
    allBeatInfo = temp_allBeatInfo;
    allocated_size = actual_beat_count;
    clear temp_allBeatInfo;
    fprintf('Memory has been shrunk to %d rows\n', allocated_size);
end

%% Delete data with NaN in allBeatInfo
fprintf('\nStarting cleanup of heartbeats with NaN values in allBeatInfo...\n');

% Get the number of records before processing - use the actual number of added heartbeats
original_beat_count = actual_beat_count;
fprintf('There are %d heartbeat records before cleanup\n', original_beat_count);

% Work directly on the original array to avoid extra memory copying
fprintf('Using memory-optimized NaN detection method...\n');

% Check for NaN values in each waveform feature point field - memory optimized version
% We focus on fields: pIndex, qIndex, sIndex, tIndex, pEnd, tEnd
has_nan = false(actual_beat_count, 1);

% Pre-allocate array for statistics
nan_counts = zeros(6, 1);  % [pIndex, qIndex, sIndex, tIndex, pEnd, tEnd]

% Single loop for NaN detection and statistics
for i = 1:actual_beat_count
    current_beat = allBeatInfo(i);
    
    % Check NaN status for each field
    is_nan_p = isnan(current_beat.pIndex);
    is_nan_q = isnan(current_beat.qIndex);
    is_nan_s = isnan(current_beat.sIndex);
    is_nan_t = isnan(current_beat.tIndex);
    is_nan_pEnd = isnan(current_beat.pEnd);
    is_nan_tEnd = isnan(current_beat.tEnd);
    
    % Count each type of NaN
    nan_counts(1) = nan_counts(1) + is_nan_p;
    nan_counts(2) = nan_counts(2) + is_nan_q;
    nan_counts(3) = nan_counts(3) + is_nan_s;
    nan_counts(4) = nan_counts(4) + is_nan_t;
    nan_counts(5) = nan_counts(5) + is_nan_pEnd;
    nan_counts(6) = nan_counts(6) + is_nan_tEnd;
    
    % Check if any NaN is present
    if is_nan_p || is_nan_q || is_nan_s || is_nan_t || is_nan_pEnd || is_nan_tEnd
        has_nan(i) = true;
    end
end

% Extract statistical results
nan_pIndex_count = nan_counts(1);
nan_qIndex_count = nan_counts(2);
nan_sIndex_count = nan_counts(3);
nan_tIndex_count = nan_counts(4);
nan_pEnd_count = nan_counts(5);
nan_tEnd_count = nan_counts(6);

fprintf('Statistics of fields containing NaN values:\n');
fprintf('  P-wave index (pIndex): %d heartbeats\n', nan_pIndex_count);
fprintf('  Q-wave index (qIndex): %d heartbeats\n', nan_qIndex_count);
fprintf('  S-wave index (sIndex): %d heartbeats\n', nan_sIndex_count);
fprintf('  T-wave index (tIndex): %d heartbeats\n', nan_tIndex_count);
fprintf('  P-wave end point (pEnd): %d heartbeats\n', nan_pEnd_count);
fprintf('  T-wave end point (tEnd): %d heartbeats\n', nan_tEnd_count);

% Get indices of records without NaN - memory optimized version
valid_indices = find(~has_nan);
invalid_count = sum(has_nan);

% If NaN records need to be deleted, use in-place compression instead of creating a new array
if invalid_count > 0
    fprintf('Using in-place compression to delete NaN records...\n');
    
    % In-place compression: move valid records to the front of the array
    write_idx = 1;
    for read_idx = 1:actual_beat_count
        if ~has_nan(read_idx)
            if write_idx ~= read_idx
                allBeatInfo(write_idx) = allBeatInfo(read_idx);
            end
            write_idx = write_idx + 1;
        end
    end
    
    % Update the actual count
    actual_beat_count = length(valid_indices);
    allocated_size = actual_beat_count;  % Update allocated size
else
    fprintf('No NaN records found, no cleanup needed\n');
end

% Clean up temporary variables
clear has_nan valid_indices nan_counts;

fprintf('A total of %d heartbeat records with NaN values were deleted (%.2f%% of the total)\n', invalid_count, (invalid_count/original_beat_count)*100);
fprintf('Remaining %d heartbeat records after cleanup\n', actual_beat_count);

%% Plot a certain heartbeat
% Ensure index is within valid range
% beat_to_plot_idx = min(10978, actual_beat_count);
% 
% fprintf('\nPlotting heartbeat #%d (from %s record %s) fs: %d Hz\n', ...
%     beat_to_plot_idx, allBeatInfo(beat_to_plot_idx).originalDatabaseName, ...
%     allBeatInfo(beat_to_plot_idx).originalRecordName, allBeatInfo(beat_to_plot_idx).fs);
% plotHeartbeat(allBeatInfo(1:actual_beat_count), beat_to_plot_idx, allBeatInfo(beat_to_plot_idx).fs);

%% Output the final count of each heartbeat type
if actual_beat_count > 0
    fprintf('\n\n--- Final Statistics of Each Heartbeat Type (based on allBeatInfo.originalAnnotationChar) ---\n');
    
    % Use only the actual valid heartbeat records
    valid_allBeatInfo = allBeatInfo(1:actual_beat_count);
    all_original_chars_final = arrayfun(@(x) x.originalAnnotationChar, valid_allBeatInfo, 'UniformOutput', false);
    
    unique_char_types_final = unique(all_original_chars_final);
    total_beats_in_allBeatInfo = actual_beat_count;
    fprintf('Total number of heartbeats processed in allBeatInfo: %d\n', total_beats_in_allBeatInfo);
    
    if total_beats_in_allBeatInfo > 0
        for char_idx = 1:length(unique_char_types_final)
            current_char = unique_char_types_final{char_idx};
            if isempty(current_char) % Handle empty character case
                count = sum(cellfun('isempty', all_original_chars_final));
                 fprintf('  Type (empty char): %d beats (%.2f%%)\n', count, (count/total_beats_in_allBeatInfo)*100);
            else
                count = sum(strcmp(all_original_chars_final, current_char));
                fprintf('  Type ''%s'': %d beats (%.2f%%)\n', current_char, count, (count/total_beats_in_allBeatInfo)*100);
            end
        end
    end
else
    fprintf('\n\nallBeatInfo is empty, no heartbeat data to count.\n');
end

%% Save allBeatInfo to a .mat file
% if actual_beat_count > 0
%     % Save only the valid heartbeat records
%     allBeatInfo = allBeatInfo(1:actual_beat_count);
%     save('results/allBeatInfo.mat', 'allBeatInfo');
%     fprintf('\n✓ allBeatInfo has been saved to results/allBeatInfo.mat (%d heartbeat records)\n', actual_beat_count);
% else
%     fprintf('\n⚠ allBeatInfo is empty, skipping save\n');
% end

%% Extract features for machine learning
trainingFeatureTable = table(); % Initialize training set feature table
testingFeatureTable = table();  % Initialize testing set feature table

% Create a struct to save the feature table for each database
all_database_features = struct();

if actual_beat_count > 0
    fprintf('\nStarting to extract features for all heartbeats, grouped by database and sampling frequency...\n');
    
    % Memory optimization: directly extract grouping information without creating a full table
    fprintf('Using memory-optimized grouping method...\n');
    db_names = cell(actual_beat_count, 1);
    fs_values = zeros(actual_beat_count, 1);
    
    for i = 1:actual_beat_count
        db_names{i} = allBeatInfo(i).originalDatabaseName;
        fs_values(i) = allBeatInfo(i).fs;
    end
    
    % Find unique database-sampling frequency combinations
    [unique_combinations, ~, group_indices] = unique(strcat(db_names, '_', string(fs_values)));
    uniqueDBFSGroups = table();
    uniqueDBFSGroups.originalDatabaseName = cell(length(unique_combinations), 1);
    uniqueDBFSGroups.fs = zeros(length(unique_combinations), 1);
    
    for i = 1:length(unique_combinations)
        first_occurrence = find(group_indices == i, 1);
        uniqueDBFSGroups.originalDatabaseName{i} = db_names{first_occurrence};
        uniqueDBFSGroups.fs(i) = fs_values{first_occurrence};
    end
    
    % Clean up temporary variables
    clear db_names fs_values unique_combinations group_indices;
    
    all_training_feature_tables = cell(height(uniqueDBFSGroups), 1);
    all_testing_feature_tables = cell(height(uniqueDBFSGroups), 1);
    
    for group_idx = 1:height(uniqueDBFSGroups)
        current_group_db_name = uniqueDBFSGroups.originalDatabaseName{group_idx};
        current_group_fs = uniqueDBFSGroups.fs(group_idx);
        
        fprintf('  Extracting feature group: Database=''%s'', fs=%d Hz\n', current_group_db_name, current_group_fs);
        
        % Filter heartbeats belonging to the current group - memory optimized version
        group_indices = false(actual_beat_count, 1);
        for beat_idx = 1:actual_beat_count
            if strcmp(allBeatInfo(beat_idx).originalDatabaseName, current_group_db_name) && ...
               allBeatInfo(beat_idx).fs == current_group_fs
                group_indices(beat_idx) = true;
            end
        end
        
        beats_for_current_group = allBeatInfo(group_indices);
        
        if isempty(beats_for_current_group)
            fprintf('    No heartbeat data in this group, skipping feature extraction.\n');
            all_training_feature_tables{group_idx} = table(); % Placeholder with empty table
            all_testing_feature_tables{group_idx} = table();  % Placeholder with empty table
            continue;
        end
        
        [featureTable_group, ~] = extractHeartbeatFeatures(beats_for_current_group, current_group_fs);
        
        % Delete rows with missing values
        if ~isempty(featureTable_group) && width(featureTable_group) > 0
            completeRows = ~any(ismissing(featureTable_group), 2);
            featureTable_group = featureTable_group(completeRows, :);
            fprintf('    Remaining %d valid heartbeat records after deleting missing values\n', height(featureTable_group));
        end
        
        % Save the feature table of each database into the struct
        fieldName = genvarname(current_group_db_name); % Generate a valid field name
        all_database_features.(fieldName) = featureTable_group;
        
        % Allocate the feature table to the training or testing set based on database allocation
        if ismember(current_group_db_name, db_allocation.training)
            all_training_feature_tables{group_idx} = featureTable_group;
            all_testing_feature_tables{group_idx} = table(); % Placeholder with empty table
            fprintf('    Extracted features for %d heartbeats for the training set.\n', height(featureTable_group));
        elseif ismember(current_group_db_name, db_allocation.testing)
            all_training_feature_tables{group_idx} = table(); % Placeholder with empty table
            all_testing_feature_tables{group_idx} = featureTable_group;
            fprintf('    Extracted features for %d heartbeats for the testing set.\n', height(featureTable_group));
        else
            fprintf('    Warning: Database ''%s'' is not specified in training or testing sets, skipping.\n', current_group_db_name);
            all_training_feature_tables{group_idx} = table(); % Placeholder with empty table
            all_testing_feature_tables{group_idx} = table();  % Placeholder with empty table
        end
        
        % Clean up temporary variables after processing each group
        clear group_indices beats_for_current_group featureTable_group completeRows fieldName;
    end
    
    % Save features of all databases and current allocation for subsequent flexible adjustments
    save('results/all_database_features.mat', 'all_database_features', 'db_allocation');
    fprintf('\nAll database features and allocation information saved to results/all_database_features.mat\n');
    
    % Merge all training feature tables
    valid_training_tables = all_training_feature_tables(~cellfun('isempty', all_training_feature_tables));
    if ~isempty(valid_training_tables)
        trainingFeatureTable = vertcat(valid_training_tables{:});
        fprintf('\nTraining set feature table has been merged. Training features extracted for a total of %d heartbeats.\n', height(trainingFeatureTable));
    else
        fprintf('\nCould not extract a valid training feature table from any group.\n');
    end
    
    % Merge all testing feature tables
    valid_testing_tables = all_testing_feature_tables(~cellfun('isempty', all_testing_feature_tables));
    if ~isempty(valid_testing_tables)
        testingFeatureTable = vertcat(valid_testing_tables{:});
        fprintf('\nTesting set feature table has been merged. Testing features extracted for a total of %d heartbeats.\n', height(testingFeatureTable));
    else
        fprintf('\nCould not extract a valid testing feature table from any group.\n');
    end
else
    fprintf('\nCould not extract heartbeat information from any record, unable to perform feature extraction.\n');
end

% %% Database Statistics Before Data Balancing
% fprintf('\n\n=== Database Statistics Before Data Balancing ===\n');
% 
% if ~isempty(trainingFeatureTable) && actual_beat_count > 0
%     fprintf('%-40s | %-8s | %-8s | %-8s | %-10s\n', 'Database Name', 'Other', 'PAC', 'PVC', 'Deleted Beats');
%     fprintf('%s\n', repmat('-', 1, 85));
% 
%     % Get all training databases
%     all_training_dbs = db_allocation.training;
%     total_other = 0;
%     total_pac = 0;
%     total_pvc = 0;
%     total_deleted = 0;
%     total_original = 0;
%     total_in_features = 0;
% 
%     for i = 1:length(all_training_dbs)
%         current_db = all_training_dbs{i};
% 
%         % Count heartbeats in allBeatInfo for the current database
%         db_beats_in_all = [];
%         for j = 1:actual_beat_count
%             if strcmp(allBeatInfo(j).originalDatabaseName, current_db)
%                 db_beats_in_all = [db_beats_in_all; allBeatInfo(j)];
%             end
%         end
% 
%         original_count = length(db_beats_in_all);
% 
%         if original_count > 0
%             % Count types of heartbeats in allBeatInfo for the current database
%             db_beat_types = {db_beats_in_all.beatType};
%             orig_other = sum(strcmp(db_beat_types, 'Other'));
%             orig_pac = sum(strcmp(db_beat_types, 'PAC'));
%             orig_pvc = sum(strcmp(db_beat_types, 'PVC'));
% 
%             % Estimate the number of heartbeats entering the feature table
%             % Since the feature table may have removed some records due to NaN, we estimate
%             % Simplified method: assume the feature table total * (original heartbeats in the database / total original heartbeats in all training databases)
% 
%             % More accurate method: check the original data corresponding to each row in the feature table
%             % However, since the feature table does not have direct database identifiers, we use proportional estimation
% 
%             % Calculate the total number of heartbeats in all training databases
%             all_training_beats = 0;
%             for k = 1:actual_beat_count
%                 if ismember(allBeatInfo(k).originalDatabaseName, all_training_dbs)
%                     all_training_beats = all_training_beats + 1;
%                 end
%             end
% 
%             % Estimate the number of heartbeats in the feature table for the current database
%             if all_training_beats > 0
%                 estimated_feature_count = round(height(trainingFeatureTable) * (original_count / all_training_beats));
%             else
%                 estimated_feature_count = 0;
%             end
% 
%             deleted_count = max(0, original_count - estimated_feature_count);
% 
%             fprintf('%-40s | %-8d | %-8d | %-8d | %-10d\n', ...
%                 current_db, orig_other, orig_pac, orig_pvc, deleted_count);
% 
%             total_other = total_other + orig_other;
%             total_pac = total_pac + orig_pac;
%             total_pvc = total_pvc + orig_pvc;
%             total_deleted = total_deleted + deleted_count;
%             total_original = total_original + original_count;
%             total_in_features = total_in_features + estimated_feature_count;
%         else
%             fprintf('%-40s | %-8s | %-8s | %-8s | %-10s\n', ...
%                 current_db, 'N/A', 'N/A', 'N/A', 'N/A');
%         end
%     end
% 
%     fprintf('%s\n', repmat('-', 1, 85));
%     fprintf('%-40s | %-8d | %-8d | %-8d | %-10d\n', ...
%         'Total', total_other, total_pac, total_pvc, total_deleted);
% 
%     fprintf('\nSupplementary Information:\n');
%     fprintf('- Total heartbeats in training databases in allBeatInfo: %d\n', total_original);
%     fprintf('- Actual number of heartbeats in training feature table: %d\n', height(trainingFeatureTable));
%     fprintf('- Estimated feature extraction success rate: %.1f%%\n', (height(trainingFeatureTable)/total_original)*100);
% 
% else
%     fprintf('Training feature table is empty or allBeatInfo has no data, cannot generate database statistics table.\n');
% end

%% Data Balancing - Adjust the quantity of each heartbeat type
fprintf('\n\n--- Performing Data Balancing, Adjusting Binary Class Heartbeat Quantities ---\n');

if ~isempty(trainingFeatureTable)
    % Get the beat type column (last column)
    beatTypes = trainingFeatureTable.BeatType;
    
    % Find indices for each class (binary classification)
    other_indices = find(strcmp(beatTypes, 'Other'));   % Other heartbeats (including original Normal, PAC, etc.)
    pvc_indices = find(strcmp(beatTypes, 'PVC'));       % PVC
    
    % Count the number of each beat type
    num_other_beats = length(other_indices);
    num_pvc_beats = length(pvc_indices);
    
    fprintf('Before balancing: Other beats = %d, PVC = %d\n', num_other_beats, num_pvc_beats);
    
    % If the number of Other beats is significantly more than PVC beats, perform balancing
    if num_other_beats > num_pvc_beats * 2
        % Calculate the number of Other beats to keep (2 times the number of PVC beats, to maintain some balance)
        target_other_count = round(num_pvc_beats * 2);
        
        % Randomly select Other beats to keep
        rng(42); % Set random seed for reproducibility
        selected_other_indices = other_indices(randperm(num_other_beats, target_other_count));
        
        % Merge the indices of the kept Other beats and all PVC beats
        selected_indices = sort([selected_other_indices; pvc_indices]);
        
        % Keep only the selected heartbeats
        trainingFeatureTable = trainingFeatureTable(selected_indices, :);
        
        fprintf('Since the number of Other beats (%d) is significantly more than PVC beats (%d)\n', num_other_beats, num_pvc_beats);
        fprintf('Randomly selected %d Other beats (2 times the number of PVC beats)\n', target_other_count);
    else
        fprintf('The number of Other beats (%d) is not significantly more than PVC beats (%d), no balancing needed\n', num_other_beats, num_pvc_beats);
    end

% Recount each beat type
    balanced_other_count = sum(strcmp(trainingFeatureTable.BeatType, 'Other'));
    balanced_pvc_count = sum(strcmp(trainingFeatureTable.BeatType, 'PVC'));
    balanced_total = height(trainingFeatureTable);
    
    fprintf('After balancing: Other beats = %d (%.1f%%), PVC = %d (%.1f%%)\n', ...
        balanced_other_count, balanced_other_count/balanced_total*100, ...
        balanced_pvc_count, balanced_pvc_count/balanced_total*100);
else
    fprintf('Training feature table is empty, cannot perform data balancing.\n');
end

%% Save training and testing feature tables
fprintf('\n=== Saving Feature Tables to File ===\n');

% Save training set feature table
if ~isempty(trainingFeatureTable) && height(trainingFeatureTable) > 0
    try
        training_save_path = 'results/trainingFeatureTable.mat';
        save(training_save_path, 'trainingFeatureTable');
        fprintf('✓ Training set feature table saved to: %s\n', training_save_path);
        fprintf('  - Contains %d heartbeat samples\n', height(trainingFeatureTable));
        fprintf('  - Contains %d features (including BeatType label)\n', width(trainingFeatureTable));
    catch ME_save_train
        fprintf('⚠ Error saving training set feature table: %s\n', ME_save_train.message);
    end
else
    fprintf('⚠ Training set feature table is empty, skipping save\n');
end

% Save testing set feature table
if ~isempty(testingFeatureTable) && height(testingFeatureTable) > 0
    try
        testing_save_path = 'results/testingFeatureTable.mat';
        save(testing_save_path, 'testingFeatureTable');
        fprintf('✓ Testing set feature table saved to: %s\n', testing_save_path);
        fprintf('  - Contains %d heartbeat samples\n', height(testingFeatureTable));
        fprintf('  - Contains %d features (including BeatType label)\n', width(testingFeatureTable));
    catch ME_save_test
        fprintf('⚠ Error saving testing set feature table: %s\n', ME_save_test.message);
    end
else
    fprintf('⚠ Testing set feature table is empty, skipping save\n');
end

fprintf('✓ Feature table saving complete\n');

%% Six Core Feature Statistical Analysis
fprintf('\n\n=== Six Core Feature Statistical Analysis ===\n');

if ~isempty(trainingFeatureTable) && height(trainingFeatureTable) > 0
    % Define the six core features (based on latest feature selection)
    core_features = {'RR_Prev', 'RR_Post', 'R_Amplitude', 'S_Amplitude', 'T_Amplitude', 'QRS_Area'};
    feature_display_names = {'Prev RR Interval', 'Post RR Interval', 'R-wave Amplitude', 'S-wave Amplitude', 'T-wave Amplitude', 'QRS Area'};
    
    % Check which features exist in the table
    available_features = {};
    available_display_names = {};
    for i = 1:length(core_features)
        if ismember(core_features{i}, trainingFeatureTable.Properties.VariableNames)
            available_features{end+1} = core_features{i};
            available_display_names{end+1} = feature_display_names{i};
        end
    end
    
    if ~isempty(available_features)
        % Get the two beat types
        beat_types = {'Other', 'PVC'};
        beat_type_names = {'Other Beats', 'PVC Beats'};
        
        % Generate feature statistics table for each beat type
        for type_idx = 1:length(beat_types)
            current_type = beat_types{type_idx};
            current_type_name = beat_type_names{type_idx};
            
            fprintf('\n--- %s Feature Statistics Table ---\n', current_type_name);
            
            % Filter data for the current beat type
            type_mask = strcmp(trainingFeatureTable.BeatType, current_type);
            type_count = sum(type_mask);
            
            if type_count == 0
                fprintf('No data for this beat type\n');
                continue;
            end
            
            fprintf('Sample Count: %d\n\n', type_count);
            fprintf('%-18s | %-10s | %-10s | %-10s | %-20s\n', 'Feature Name', 'Mean', 'Std Dev', 'Valid Rate', 'Range');
            fprintf('%s\n', repmat('-', 1, 80));
            
            % Calculate statistics for each feature
            for feat_idx = 1:length(available_features)
                feature_name = available_features{feat_idx};
                display_name = available_display_names{feat_idx};
                
                % Get data for the current type and feature
                feature_data = trainingFeatureTable.(feature_name)(type_mask);
                
                % Remove NaN values
                valid_data = feature_data(~isnan(feature_data));
                
                if ~isempty(valid_data)
                    % Calculate statistics
                    mean_val = mean(valid_data);
                    std_val = std(valid_data);
                    valid_rate = (length(valid_data) / length(feature_data)) * 100;
                    min_val = min(valid_data);
                    max_val = max(valid_data);
                    
                    fprintf('%-18s | %-10.3f | %-10.3f | %-9.1f%% | [%.3f,%.3f]\n', ...
                        display_name, mean_val, std_val, valid_rate, min_val, max_val);
                else
                    fprintf('%-18s | %-10s | %-10s | %-9s | %-20s\n', ...
                        display_name, 'N/A', 'N/A', 'N/A', 'N/A');
                end
            end
        end
        
        % Add overall feature comparison table
        fprintf('\n--- Overall Feature Comparison Table for Both Beat Types ---\n');
        fprintf('%-18s | %-12s | %-12s | %-12s | %-12s\n', ...
            'Feature Name', 'Other Mean', 'Other Std Dev', 'PVC Mean', 'PVC Std Dev');
        fprintf('%s\n', repmat('-', 1, 75));
        
        for feat_idx = 1:length(available_features)
            feature_name = available_features{feat_idx};
            display_name = available_display_names{feat_idx};
            
            % Calculate statistics for both types
            stats_row = {display_name};
            
            for type_idx = 1:length(beat_types)
                current_type = beat_types{type_idx};
                type_mask = strcmp(trainingFeatureTable.BeatType, current_type);
                feature_data = trainingFeatureTable.(feature_name)(type_mask);
                valid_data = feature_data(~isnan(feature_data));
                
                if ~isempty(valid_data)
                    mean_val = mean(valid_data);
                    std_val = std(valid_data);
                    stats_row{end+1} = sprintf('%.3f', mean_val);
                    stats_row{end+1} = sprintf('%.3f', std_val);
                else
                    stats_row{end+1} = 'N/A';
                    stats_row{end+1} = 'N/A';
                end
            end
            
            fprintf('%-18s | %-12s | %-12s | %-12s | %-12s\n', ...
                stats_row{:});
        end
        
    else
        fprintf('The six core features were not found in the training feature table, cannot generate statistics table.\n');
        fprintf('Available features list: %s\n', strjoin(trainingFeatureTable.Properties.VariableNames, ', '));
    end
else
    fprintf('Training feature table is empty or has no data, cannot generate feature statistics table.\n');
end

%% Train Classifier
fprintf('\n\n--- Training Classifier ---\n');

if height(trainingFeatureTable) > 0
    % Training set is ready, start training the classifier
    fprintf('Training classifier with %d heartbeat samples\n', height(trainingFeatureTable));
    
    % Train the classifier
    [trainedClassifier, validationAccuracy] = trainClassifier(trainingFeatureTable);
    
    fprintf('Classifier training complete. Cross-validation accuracy: %.2f%%\n', validationAccuracy * 100);
    
    %% Evaluate classifier performance on the test set
    if height(testingFeatureTable) > 0
        fprintf('\n\n--- Evaluating Classifier Performance on the Test Set ---\n');
        fprintf('Test set contains %d heartbeat samples\n', height(testingFeatureTable));
        
        % Use the trained classifier to predict on the test set
        [predicted_labels, prediction_scores] = trainedClassifier.predictFcn(testingFeatureTable);
        
        % Calculate overall accuracy
        actual_labels = testingFeatureTable.BeatType;
        
        % Handle comparison issues with cell arrays
        if iscell(predicted_labels) && iscell(actual_labels)
            % Both are cell arrays, use strcmp
            accuracy = sum(strcmp(predicted_labels, actual_labels)) / length(actual_labels) * 100;
        elseif iscell(predicted_labels) && ~iscell(actual_labels)
            % predicted_labels is a cell, actual_labels is not, convert and compare
            accuracy = sum(strcmp(predicted_labels, cellstr(actual_labels))) / length(actual_labels) * 100;
        elseif ~iscell(predicted_labels) && iscell(actual_labels)
            % predicted_labels is not a cell, actual_labels is, convert and compare
            accuracy = sum(strcmp(cellstr(predicted_labels), actual_labels)) / length(actual_labels) * 100;
        else
            % Neither are cell arrays, use original comparison method
            accuracy = sum(predicted_labels == actual_labels) / length(actual_labels) * 100;
        end
        
        fprintf('Overall accuracy on the test set: %.2f%%\n', accuracy);
        
        % Calculate confusion matrix
        % Ensure class_names is a cell array
        if iscell(actual_labels)
            class_names = unique(actual_labels);
        else
            class_names = categories(categorical(actual_labels));
        end
        num_classes = length(class_names);
        confusion_mat = zeros(num_classes, num_classes);
        
        % Ensure labels are all in cell array format for unified processing
        if ~iscell(actual_labels)
            actual_labels = cellstr(actual_labels);
        end
        if ~iscell(predicted_labels)
            predicted_labels = cellstr(predicted_labels);
        end
        
        for i = 1:num_classes
            for j = 1:num_classes
                confusion_mat(i, j) = sum(strcmp(actual_labels, class_names{i}) & strcmp(predicted_labels, class_names{j}));
            end
        end
        
        % Display confusion matrix (absolute numbers)
        fprintf('\n=== Confusion Matrix (Absolute Numbers) ===\n');
        % Create table header
        header = sprintf('%-12s', 'Actual\Pred');
        for i = 1:num_classes
            header = [header, sprintf('%12s', class_names{i})];
        end
        fprintf('%s\n', header);
        fprintf('%s\n', repmat('-', 1, length(header)));
        
        % Display matrix content
        for i = 1:num_classes
            row_str = sprintf('%-12s', class_names{i});
            for j = 1:num_classes
                row_str = [row_str, sprintf('%12d', confusion_mat(i, j))];
            end
            fprintf('%s\n', row_str);
        end
        
        % Calculate and display percentage confusion matrix
        fprintf('\n=== Confusion Matrix (Row Percentages) ===\n');
        % Calculate sum of each row
        row_sums = sum(confusion_mat, 2);
        
        % Create table header
        header = sprintf('%-12s', 'Actual\Pred');
        for i = 1:num_classes
            header = [header, sprintf('%12s', class_names{i})];
        end
        fprintf('%s\n', header);
        fprintf('%s\n', repmat('-', 1, length(header)));
        
        % Display percentage matrix content
        for i = 1:num_classes
            row_str = sprintf('%-12s', class_names{i});
            for j = 1:num_classes
                if row_sums(i) > 0
                    percentage = confusion_mat(i, j) / row_sums(i) * 100;
                    row_str = [row_str, sprintf('%11.1f%%', percentage)];
                else
                    row_str = [row_str, sprintf('%11s', 'N/A')];
                end
            end
            fprintf('%s\n', row_str);
        end
        
        % Calculate performance metrics for each class
        fprintf('\n=== Performance Metrics for Each Class ===\n');
        
        % Pre-allocate TPR and FNR matrix
        tpr_fnr_matrix = zeros(num_classes, 2); % Column 1: TPR, Column 2: FNR
        
        for i = 1:num_classes
            class_name = class_names{i};
            
            % True Positive, False Positive, False Negative, and True Negative
            TP = confusion_mat(i, i);
            FP = sum(confusion_mat(:, i)) - TP;
            FN = sum(confusion_mat(i, :)) - TP;
            TN = sum(confusion_mat(:)) - TP - FP - FN;
            
            % Calculate metrics
            if (TP + FN) > 0
                sensitivity = TP / (TP + FN) * 100; % Recall/Sensitivity = TPR
                tpr_fnr_matrix(i, 1) = sensitivity;
                tpr_fnr_matrix(i, 2) = 100 - sensitivity; % FNR = 100 - TPR
            else
                sensitivity = 0;
                tpr_fnr_matrix(i, 1) = 0;
                tpr_fnr_matrix(i, 2) = 0;
            end
            
            if (TN + FP) > 0
                specificity = TN / (TN + FP) * 100; % Specificity
            else
                specificity = 0;
            end
            
            if (TP + FP) > 0
                precision = TP / (TP + FP) * 100;   % Precision
            else
                precision = 0;
            end
            
            if (precision + sensitivity) > 0
                f1_score = 2 * (precision * sensitivity) / (precision + sensitivity); % F1 Score
            else
                f1_score = 0;
            end
            
            fprintf('Class %s:\n', class_name);
            fprintf('  True Positive Rate (TPR)/Sensitivity/Recall: %.2f%%\n', sensitivity);
            fprintf('  False Negative Rate (FNR): %.2f%%\n', 100 - sensitivity);
            fprintf('  Specificity: %.2f%%\n', specificity);
            fprintf('  Precision: %.2f%%\n', precision);
            fprintf('  F1 Score: %.2f\n', f1_score);
            fprintf('  Sample counts: TP=%d, FP=%d, FN=%d, TN=%d\n\n', TP, FP, FN, TN);
        end
        
        % Display TPR and FNR summary table
        fprintf('=== TPR and FNR Summary Table ===\n');
        fprintf('%-15s%12s%12s\n', 'Class', 'TPR(%)', 'FNR(%)');
        fprintf('%s\n', repmat('-', 1, 39));
        for i = 1:num_classes
            fprintf('%-15s%11.2f%12.2f\n', class_names{i}, tpr_fnr_matrix(i, 1), tpr_fnr_matrix(i, 2));
        end
        
        % Save model and test results
        results_struct = struct();
        results_struct.trainedClassifier = trainedClassifier;
        results_struct.validationAccuracy = validationAccuracy;
        results_struct.testAccuracy = accuracy/100;
        results_struct.confusionMatrix = confusion_mat;
        results_struct.classNames = class_names;
        results_struct.predictionScores = prediction_scores;
        results_struct.predictedLabels = predicted_labels;
        results_struct.actualLabels = actual_labels;
        results_struct.tprFnrMatrix = tpr_fnr_matrix; % Add TPR/FNR matrix
                
        % Visualize confusion matrix
        figure('Name', 'Heartbeat Classification Confusion Matrix', 'NumberTitle', 'off');
        confusionchart(confusion_mat, class_names);
        title('Heartbeat Classification Confusion Matrix (Absolute Numbers)');
        
        % Create percentage confusion matrix plot
        figure('Name', 'Heartbeat Classification Confusion Matrix (Percentages)', 'NumberTitle', 'off');
        percentage_matrix = confusion_mat ./ repmat(sum(confusion_mat, 2), 1, num_classes) * 100;
        percentage_matrix(isnan(percentage_matrix)) = 0; % Handle division by zero
        
        % Use a heatmap to display the percentage confusion matrix
        imagesc(percentage_matrix);
        colormap('bone');
        colorbar;
        
        % Set axis labels
        set(gca, 'XTick', 1:num_classes, 'XTickLabel', class_names);
        set(gca, 'YTick', 1:num_classes, 'YTickLabel', class_names);
        xlabel('Predicted Class');
        ylabel('Actual Class');
        title('Heartbeat Classification Confusion Matrix (Row Percentages)');
        
        % Add percentage text to each cell
        for i = 1:num_classes
            for j = 1:num_classes
                text(j, i, sprintf('%.1f%%', percentage_matrix(i, j)), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'Color', 'white', 'FontWeight', 'bold');
            end
        end
        
        % Create TPR/FNR visualization chart
        figure('Name', 'TPR and FNR Comparison', 'NumberTitle', 'off');
        x = 1:num_classes;
        bar_width = 0.35;
        
        bar(x - bar_width/2, tpr_fnr_matrix(:, 1), bar_width, 'FaceColor', [0.2 0.7 0.2], 'DisplayName', 'TPR (%)');
        hold on;
        bar(x + bar_width/2, tpr_fnr_matrix(:, 2), bar_width, 'FaceColor', [0.7 0.2 0.2], 'DisplayName', 'FNR (%)');
        
        xlabel('Heartbeat Class');
        ylabel('Percentage (%)');
        title('TPR and FNR Comparison for Each Class');
        legend('Location', 'best');
        set(gca, 'XTick', x, 'XTickLabel', class_names);
        grid on;
        ylim([0, 100]);
        
        % Add numerical labels to the bar chart
        for i = 1:num_classes
            text(i - bar_width/2, tpr_fnr_matrix(i, 1) + 2, sprintf('%.1f', tpr_fnr_matrix(i, 1)), ...
                'HorizontalAlignment', 'center', 'FontSize', 9);
            text(i + bar_width/2, tpr_fnr_matrix(i, 2) + 2, sprintf('%.1f', tpr_fnr_matrix(i, 2)), ...
                'HorizontalAlignment', 'center', 'FontSize', 9);
        end
        
    else
        fprintf('Test set is empty, cannot evaluate classifier performance.\n');
    end
else
    fprintf('Training feature table is empty, cannot train classifier.\n');
end

%% Final Memory Cleanup
fprintf('\n=== Performing Final Memory Cleanup ===\n');

% Clean up large temporary variables
clear all_training_feature_tables all_testing_feature_tables;
clear uniqueDBFSGroups all_database_features;

% Optionally clean up the complete allBeatInfo if no longer needed
% clear allBeatInfo;  % Uncomment if you no longer need it for subsequent use

fprintf('Memory cleanup complete\n');
fprintf('\nProcessing complete.\n');

% Hint to the user on how to use the switchDBAllocation function
fprintf('\n=== Usage Hints ===\n');
fprintf('1. To switch database allocation, use:\n');
fprintf('   [newTrainTable, newTestTable, newAlloc] = switchDBAllocation(''DatabaseName'', ''training'' or ''testing'')\n');
fprintf('   Example: [trainData, testData, ~] = switchDBAllocation(''MIT_data'', ''testing'')\n');
fprintf('\n2. Memory optimization hints:\n');
fprintf('   - If you no longer need the complete heartbeat information, you can run: clear allBeatInfo\n');
fprintf('   - Run `clear all` after processing to release all memory\n');
fprintf('   - It is recommended to close unnecessary figure windows when processing large datasets\n\n');



