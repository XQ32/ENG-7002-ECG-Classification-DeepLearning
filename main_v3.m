%% Main function: Load ECG data from .mat files, detect heartbeats, classify, and extract features for machine learning
% This program uses loadECGData to load data, ecgFilter for filtering
% Then it uses detectAndClassifyHeartbeats to detect and classify heartbeats, and extract features for machine learning
clc;
clear;
close all;

%% Parameter Settings

% Database allocation configuration - easily modifiable parameters to specify which databases are used for training/testing
db_allocation = struct();
% Databases for training
db_allocation.training = {'MIT_data', 'INCART_data', 'european_st_t_database_1_0_0_data', ...
    'mit_bih_long_term_ecg_database_1_0_0_data'};       
% Databases for testing
db_allocation.testing = {'leipzig_heart_center_ecg_database_arrhythmias_in_children_and_p'};  


% MAT database file list (assumed to be in the 'results' folder)
mat_files_info = {
    struct('fileName', 'MIT_data.mat', 'structName', 'MIT_data', 'powerLineFreq', 60), ...
    struct('fileName', 'INCART_data.mat', 'structName', 'INCART_data', 'powerLineFreq', 50), ...
    struct('fileName', 'european-st-t-database-1.0.0_data.mat', 'structName', 'european_st_t_database_1_0_0_data', 'powerLineFreq', 50), ...
    struct('fileName', 'mit-bih-long-term-ecg-database-1.0.0_data.mat', 'structName', 'mit_bih_long_term_ecg_database_1_0_0_data', 'powerLineFreq', 60), ...
    struct('fileName', 'leipzig-heart-center-ecg-database-arrhythmias-in-children-and-patients-with-congenital-heart-disease-1.0.0_data.mat', 'structName', 'leipzig_heart_center_ecg_database_arrhythmias_in_children_and_p', 'powerLineFreq', 50), ...
    % struct('fileName', 'sudden-cardiac-death-holter-database-1.0.0_data.mat', 'structName', 'sudden_cardiac_death_holter_database_1_0_0_data', 'powerLineFreq', 60), ...
    % struct('fileName', 'shdb-af-a-japanese-holter-ecg-database-of-atrial-fibrillation-1.0.1_data.mat', 'structName', 'shdb_af_a_japanese_holter_ecg_database_of_atrial_fibrillation_1_0_1_data', 'powerLineFreq', 50), ...

};
results_dir = 'results'; % Directory where .mat files are located

%% Initialize Variables
% allHeartbeatSegments = cell(0, 1);
% Preallocate memory for 3 million rows for the allBeatInfo struct array
initial_struct = struct('beatType', 0, 'segment', [], 'rIndex', 0, 'pIndex', NaN, ...
                'qIndex', NaN, 'sIndex', NaN, 'tIndex', NaN, 'pEnd', NaN, 'tEnd', NaN, ...
                'segmentStartIndex', 0, 'fs', 0, 'originalRecordName', '', ...
                'originalDatabaseName', '', 'originalAnnotationChar', '');
                
allBeatInfo = repmat(initial_struct, 10000000, 1);  % Preallocate 10 million rows
actual_beat_count = 0;  % Track the number of actually added heartbeats

%% Process data records one by one
fprintf('Starting to process database .mat files, detect heartbeats and extract features...\n');

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
        fprintf('Warning: Expected struct variable ''%s'' not found in %s. Skipping this database.\n', current_db_info.structName, current_db_info.fileName);
        continue;
    end
    
    db_records_array = loaded_data.(current_db_info.structName);
    fprintf('Database %s contains %d records.\n', current_db_info.structName, length(db_records_array));

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
             fprintf('    Warning: Annotations or annotation types for record %s are empty. Skipping this record.\n', recordName);
            continue;
        end
        
        ATRTIMED = double(current_record.annotations) / fs;
        ANNOT_CHAR_orig = current_record.anntype; % Character annotations

        % Convert character ANNOT_CHAR_orig to numeric ANNOTD
        ANNOTD = zeros(size(ANNOT_CHAR_orig, 1), 1);
        for k_ann = 1:size(ANNOT_CHAR_orig, 1)
            ann_char_single = strtrim(ANNOT_CHAR_orig(k_ann,:)); 
            switch upper(ann_char_single)
                case 'N'
                    ANNOTD(k_ann) = 1; % Normal like
                case 'V'
                    ANNOTD(k_ann) = 5; % PVC
                case 'A'
                    ANNOTD(k_ann) = 8; % PAC/SVE like
                % Other types default to 0, detectAndClassifyHeartbeats will map them to 1 (Normal)
            end
        end
        
        % Use try-catch to ensure the program continues even if detectAndClassifyHeartbeats errors out
        try
            % Call ecgFilter based on the database's power line frequency parameter
            power_line_freq = current_db_info.powerLineFreq;
            fprintf('    Filtering %d Hz power line interference using a filter\n', power_line_freq);
            [ecg_filtered, filter_info] = ecgFilter(M, fs, 2, power_line_freq);
            
            % Optional: If filter information needs to be output to the console
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
                continue;
            end

            % Add original character annotations and source information to beatInfo_rec_numeric
            if ~isempty(beatInfo_rec_numeric)
                tempBeatInfoArray = repmat(struct(...
                    'beatType', 0, 'segment', [], 'rIndex', 0, 'pIndex', NaN, ...
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
                    if abs(double(current_record.annotations(closest_idx_in_orig_ann)) - r_peak_in_original_ecg) <= 2 % Allow +/-2 samples error
                        tempBeatInfoArray(j).originalAnnotationChar = strtrim(ANNOT_CHAR_orig(closest_idx_in_orig_ann,:));
                    else
                        tempBeatInfoArray(j).originalAnnotationChar = '?'; % No exact match found
                        fprintf('    Warning: Record %s, R-peak@%d could not be exactly matched back to original character annotation. Closest distance %d sample points.\n', ...
                                recordName, r_peak_in_original_ecg, ...
                                abs(double(current_record.annotations(closest_idx_in_orig_ann)) - r_peak_in_original_ecg));
                    end
                end
                beatInfo_rec = tempBeatInfoArray;
            else
                beatInfo_rec = [];
            end

            % allHeartbeatSegments = [allHeartbeatSegments; heartbeatSegments_rec];
            allBeatInfo(actual_beat_count + 1:actual_beat_count + length(beatInfo_rec)) = beatInfo_rec;
            actual_beat_count = actual_beat_count + length(beatInfo_rec);
    
            if ~isempty(beatInfo_rec)
                types_char = arrayfun(@(x) x.originalAnnotationChar, beatInfo_rec, 'UniformOutput', false);
                numNormal_char = sum(ismember(types_char, {'N', 'L', 'R', '/'}));
                numPVC_char    = sum(strcmp(types_char, 'V'));
                numPAC_char    = sum(ismember(types_char, {'A', 'S'}));
                numUnknown_char= sum(strcmp(types_char, '?'));
                
                fprintf('  Record %s processed. Extracted heartbeats (character statistics): Normal-like=%d, PVC=%d, PAC=%d, Unmatched characters=%d\n', ...
                    recordName, numNormal_char, numPVC_char, numPAC_char, numUnknown_char);
                fprintf('**************************************************************************\n');
            else
                fprintf('  No heartbeats extracted from record %s.\n', recordName);
            end
        catch ME_processing
            fprintf('  Error processing record %s: %s\n  Skipping this record and continuing.\n', recordName, ME_processing.message);
            continue;
        end
    end
end

%% Remove data with NaN values from allBeatInfo
fprintf('\nStarting to clean up heartbeats with NaN values in allBeatInfo...\n');

% Get the number of records before processing - use the actual number of added heartbeats
original_beat_count = actual_beat_count;
fprintf('There are %d heartbeat records before cleaning\n', original_beat_count);

% Only check the actually added heartbeat records
working_allBeatInfo = allBeatInfo(1:actual_beat_count);

% Check if each waveform feature point field contains NaN values
% Fields we are interested in include pIndex, qIndex, sIndex, tIndex, pEnd, tEnd
has_nan = false(size(working_allBeatInfo));
for i = 1:length(working_allBeatInfo)
    if isnan(working_allBeatInfo(i).pIndex) || isnan(working_allBeatInfo(i).qIndex) || ...
       isnan(working_allBeatInfo(i).sIndex) || isnan(working_allBeatInfo(i).tIndex) || ...
       isnan(working_allBeatInfo(i).pEnd) || isnan(working_allBeatInfo(i).tEnd)
        has_nan(i) = true;
    end
end

% Count the number of NaN values for each type
nan_pIndex_count = sum(arrayfun(@(x) isnan(x.pIndex), working_allBeatInfo));
nan_qIndex_count = sum(arrayfun(@(x) isnan(x.qIndex), working_allBeatInfo));
nan_sIndex_count = sum(arrayfun(@(x) isnan(x.sIndex), working_allBeatInfo));
nan_tIndex_count = sum(arrayfun(@(x) isnan(x.tIndex), working_allBeatInfo));
nan_pEnd_count = sum(arrayfun(@(x) isnan(x.pEnd), working_allBeatInfo));
nan_tEnd_count = sum(arrayfun(@(x) isnan(x.tEnd), working_allBeatInfo));

fprintf('Statistics of fields with NaN values:\n');
fprintf('  P-wave index (pIndex): %d heartbeats\n', nan_pIndex_count);
fprintf('  Q-wave index (qIndex): %d heartbeats\n', nan_qIndex_count);
fprintf('  S-wave index (sIndex): %d heartbeats\n', nan_sIndex_count);
fprintf('  T-wave index (tIndex): %d heartbeats\n', nan_tIndex_count);
fprintf('  P-wave end point (pEnd): %d heartbeats\n', nan_pEnd_count);
fprintf('  T-wave end point (tEnd): %d heartbeats\n', nan_tEnd_count);

% Get indices of records without NaN
valid_indices = find(~has_nan);
invalid_count = sum(has_nan);

% Keep records without NaN
allBeatInfo_cleaned = working_allBeatInfo(valid_indices);
actual_beat_count = length(allBeatInfo_cleaned);

% Reset allBeatInfo and fill with cleaned data
allBeatInfo = repmat(initial_struct, 3000000, 1);
allBeatInfo(1:actual_beat_count) = allBeatInfo_cleaned;

fprintf('A total of %d heartbeat records with NaN values were deleted (%.2f%% of the total)\n', invalid_count, (invalid_count/original_beat_count)*100);
fprintf('Remaining %d heartbeat records after cleaning\n', actual_beat_count);

%% Plot a certain heartbeat
% Ensure the index is within the valid range
beat_to_plot_idx = min(10978, actual_beat_count);

fprintf('\nPlotting heartbeat %d (from %s record %s) fs: %d Hz\n', ...
    beat_to_plot_idx, allBeatInfo(beat_to_plot_idx).originalDatabaseName, ...
    allBeatInfo(beat_to_plot_idx).originalRecordName, allBeatInfo(beat_to_plot_idx).fs);
plotHeartbeat(allBeatInfo(1:actual_beat_count), beat_to_plot_idx, allBeatInfo(beat_to_plot_idx).fs);

%% Output final counts of each type of heartbeat
if actual_beat_count > 0
    fprintf('\n\n--- Final counts of each type of heartbeat (based on allBeatInfo.originalAnnotationChar) ---\n');
    
    % Only use actually valid heartbeat records
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
                 fprintf('  Type (empty character): %d beats (%.2f%%)\n', count, (count/total_beats_in_allBeatInfo)*100);
            else
                count = sum(strcmp(all_original_chars_final, current_char));
                fprintf('  Type ''%s'': %d beats (%.2f%%)\n', current_char, count, (count/total_beats_in_allBeatInfo)*100);
            end
        end
    end
else
    fprintf('\n\nallBeatInfo is empty, no heartbeat data to count.\n');
end

%% Extract features for machine learning
trainingFeatureTable = table(); % Initialize training feature table
testingFeatureTable = table();  % Initialize testing feature table

if actual_beat_count > 0
    fprintf('\nStarting to group all heartbeats by database and sampling frequency to extract features...\n');
    
    % Only use actually valid heartbeat records
    valid_allBeatInfo = allBeatInfo(1:actual_beat_count);
    
    tempInfoForGrouping = struct2table(valid_allBeatInfo);
    uniqueDBFSGroups = unique(tempInfoForGrouping(:, {'originalDatabaseName', 'fs'}), 'rows');
    
    all_training_feature_tables = cell(height(uniqueDBFSGroups), 1);
    all_testing_feature_tables = cell(height(uniqueDBFSGroups), 1);
    
    for group_idx = 1:height(uniqueDBFSGroups)
        current_group_db_name = uniqueDBFSGroups.originalDatabaseName{group_idx};
        current_group_fs = uniqueDBFSGroups.fs(group_idx);
        
        fprintf('  Extracting feature group: Database=''%s'', fs=%d Hz\n', current_group_db_name, current_group_fs);
        
        % Filter heartbeats belonging to the current group
        group_indices = find(strcmp({valid_allBeatInfo.originalDatabaseName}, current_group_db_name) & ([valid_allBeatInfo.fs] == current_group_fs));
        beats_for_current_group = valid_allBeatInfo(group_indices);
        
        if isempty(beats_for_current_group)
            fprintf('    No heartbeat data for this group, skipping feature extraction.\n');
            all_training_feature_tables{group_idx} = table(); % Empty table placeholder
            all_testing_feature_tables{group_idx} = table();  % Empty table placeholder
            continue;
        end
        
        [featureTable_group, ~] = extractHeartbeatFeatures(beats_for_current_group, current_group_fs);
        
        % Assign feature table to training or testing set based on database allocation
        if ismember(current_group_db_name, db_allocation.training)
            all_training_feature_tables{group_idx} = featureTable_group;
            all_testing_feature_tables{group_idx} = table(); % Empty table placeholder
            fprintf('    Extracted features for %d heartbeats for training set.\n', height(featureTable_group));
        elseif ismember(current_group_db_name, db_allocation.testing)
            all_training_feature_tables{group_idx} = table(); % Empty table placeholder
            all_testing_feature_tables{group_idx} = featureTable_group;
            fprintf('    Extracted features for %d heartbeats for testing set.\n', height(featureTable_group));
        else
            fprintf('    Warning: Database ''%s'' not specified in training or testing set, skipping.\n', current_group_db_name);
            all_training_feature_tables{group_idx} = table(); % Empty table placeholder
            all_testing_feature_tables{group_idx} = table();  % Empty table placeholder
        end
    end
    
    % Combine all training feature tables
    valid_training_tables = all_training_feature_tables(~cellfun('isempty', all_training_feature_tables));
    if ~isempty(valid_training_tables)
        trainingFeatureTable = vertcat(valid_training_tables{:});
        fprintf('\nTraining feature table combined. Extracted training features for a total of %d heartbeats.\n', height(trainingFeatureTable));
    else
        fprintf('\nFailed to extract valid training feature table from any group.\n');
    end
    
    % Combine all testing feature tables
    valid_testing_tables = all_testing_feature_tables(~cellfun('isempty', all_testing_feature_tables));
    if ~isempty(valid_testing_tables)
        testingFeatureTable = vertcat(valid_testing_tables{:});
        fprintf('\nTesting feature table combined. Extracted testing features for a total of %d heartbeats.\n', height(testingFeatureTable));
    else
        fprintf('\nFailed to extract valid testing feature table from any group.\n');
    end

    % Remove rows with missing values
    if ~isempty(trainingFeatureTable) && width(trainingFeatureTable) > 0
        completeRows = ~any(ismissing(trainingFeatureTable), 2);
        trainingFeatureTable = trainingFeatureTable(completeRows, :);
        fprintf('After removing missing values, %d valid heartbeat records remain in the training set\n', height(trainingFeatureTable));
    end
    
    if ~isempty(testingFeatureTable) && width(testingFeatureTable) > 0
        completeRows = ~any(ismissing(testingFeatureTable), 2);
        testingFeatureTable = testingFeatureTable(completeRows, :);
        fprintf('After removing missing values, %d valid heartbeat records remain in the testing set\n', height(testingFeatureTable));
    end
else
    fprintf('\nFailed to extract heartbeat information from any record, unable to perform feature extraction.\n');
end

%% Data balancing - Reduce the number of normal heartbeats to approximately match the number of abnormal heartbeats
fprintf('\n\n--- Performing data balancing to reduce the number of normal heartbeats ---\n');

if ~isempty(trainingFeatureTable)
    % Get the heartbeat type column (last column)
    beatTypes = trainingFeatureTable.BeatType;
    
    % Find indices of each type of heartbeat
    normal_indices = find(beatTypes == '1');  % Normal heartbeats (Type 1)
    pvc_indices = find(beatTypes == '5');     % PVC (Type 5)
    pac_indices = find(beatTypes == '8');     % PAC (Type 8)
    
    % Count the number of each type of heartbeat
    num_normal_beats = length(normal_indices);
    num_pvc_beats = length(pvc_indices);
    num_pac_beats = length(pac_indices);
    
    fprintf('Before balancing: Normal heartbeats = %d, PVC = %d, PAC = %d\n', num_normal_beats, num_pvc_beats, num_pac_beats);
    
    % Calculate the total number of abnormal heartbeats
    total_abnormal_beats = num_pvc_beats + num_pac_beats;
    fprintf('Total number of abnormal heartbeats = %d\n', total_abnormal_beats);
    
    % If the number of normal heartbeats is significantly higher than abnormal heartbeats, allow slightly more normal heartbeats (20% margin)
    if num_normal_beats > total_abnormal_beats * 1.5
        % Calculate the number of normal heartbeats to retain (1.2 times the number of abnormal heartbeats)
        target_normal_count = round(total_abnormal_beats * 1.5);
        
        % Randomly select normal heartbeats to retain
        rng(42); % Set random seed for reproducibility
        selected_normal_indices = normal_indices(randperm(num_normal_beats, target_normal_count));
        
        % Combine retained normal heartbeats and all abnormal heartbeat indices
        selected_indices = sort([selected_normal_indices; pvc_indices; pac_indices]);
        
        % Retain only the selected heartbeats
        trainingFeatureTable = trainingFeatureTable(selected_indices, :);
        
        fprintf('Since the number of normal heartbeats (%d) is significantly higher than the number of abnormal heartbeats (%d)\n', num_normal_beats, total_abnormal_beats);
        fprintf('Randomly selected %d normal heartbeats (1.5 times the total number of abnormal heartbeats)\n', target_normal_count);
    else
        fprintf('The number of normal heartbeats (%d) is not significantly higher than the number of abnormal heartbeats (%d), no balancing needed\n', num_normal_beats, total_abnormal_beats);
    end
    
    % Recount the number of each type of heartbeat
    balanced_normal_count = sum(trainingFeatureTable.BeatType == '1');
    balanced_pvc_count = sum(trainingFeatureTable.BeatType == '5');
    balanced_pac_count = sum(trainingFeatureTable.BeatType == '8');
    balanced_total = height(trainingFeatureTable);
    
    fprintf('After balancing: Normal heartbeats = %d (%.1f%%), PVC = %d (%.1f%%), PAC = %d (%.1f%%)\n', ...
        balanced_normal_count, balanced_normal_count/balanced_total*100, ...
        balanced_pvc_count, balanced_pvc_count/balanced_total*100, ...
        balanced_pac_count, balanced_pac_count/balanced_total*100);
else
    fprintf('Training feature table is empty, unable to perform data balancing.\n');
end

%% Split training and testing sets - Leave 10% of each category as the testing set
fprintf('\n\n--- Train the classifier ---\n');

if height(trainingFeatureTable) > 0
    % Training set is ready, start training the classifier
    fprintf('Using %d heartbeat samples to train the classifier\n', height(trainingFeatureTable));
    
    % Train the classifier
    [trainedClassifier, validationAccuracy] = trainClassifier(trainingFeatureTable);
    
    fprintf('Classifier training complete. Cross-validation accuracy: %.2f%%\n', validationAccuracy * 100);
    
    %% Evaluate classifier performance using the testing set
    if height(testingFeatureTable) > 0
        fprintf('\n\n--- Evaluate classifier performance on the testing set ---\n');
        fprintf('The testing set contains %d heartbeat samples\n', height(testingFeatureTable));
        
        % Use the trained classifier to predict the testing set
        [predicted_labels, prediction_scores] = trainedClassifier.predictFcn(testingFeatureTable);
        
        % Calculate overall accuracy
        actual_labels = testingFeatureTable.BeatType;
        accuracy = sum(predicted_labels == actual_labels) / length(actual_labels) * 100;
        fprintf('Overall accuracy on the testing set: %.2f%%\n', accuracy);
        
        % Calculate confusion matrix
        class_names = categories(categorical(actual_labels));
        num_classes = length(class_names);
        confusion_mat = zeros(num_classes, num_classes);
        
        for i = 1:num_classes
            for j = 1:num_classes
                confusion_mat(i, j) = sum((actual_labels == class_names{i}) & (predicted_labels == class_names{j}));
            end
        end
        
        % Display confusion matrix
        fprintf('\nConfusion matrix:\n');
        % Create table header
        header = 'Actual\\Predicted';
        for i = 1:num_classes
            header = [header, sprintf('\t%s', class_names{i})];
        end
        fprintf('%s\n', header);
        
        % Display matrix content
        for i = 1:num_classes
            row_str = class_names{i};
            for j = 1:num_classes
                row_str = [row_str, sprintf('\t%d', confusion_mat(i, j))];
            end
            fprintf('%s\n', row_str);
        end
        
        % Calculate performance metrics for each category
        fprintf('\nPerformance metrics for each category:\n');
        for i = 1:num_classes
            class_name = class_names{i};
            
            % True positive, false positive, false negative, and true negative
            TP = confusion_mat(i, i);
            FP = sum(confusion_mat(:, i)) - TP;
            FN = sum(confusion_mat(i, :)) - TP;
            TN = sum(confusion_mat(:)) - TP - FP - FN;
            
            % Calculate metrics
            sensitivity = TP / (TP + FN) * 100; % Recall/Sensitivity
            specificity = TN / (TN + FP) * 100; % Specificity
            precision = TP / (TP + FP) * 100;   % Precision
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity); % F1 score
            
            fprintf('Category %s:\n', class_name);
            fprintf('  Sensitivity/Recall: %.2f%%\n', sensitivity);
            fprintf('  Specificity: %.2f%%\n', specificity);
            fprintf('  Precision: %.2f%%\n', precision);
            fprintf('  F1 score: %.2f\n', f1_score);
        end
        
        % Save model and testing results
        results_struct = struct();
        results_struct.trainedClassifier = trainedClassifier;
        results_struct.validationAccuracy = validationAccuracy;
        results_struct.testAccuracy = accuracy/100;
        results_struct.confusionMatrix = confusion_mat;
        results_struct.classNames = class_names;
        results_struct.predictionScores = prediction_scores;
        results_struct.predictedLabels = predicted_labels;
        results_struct.actualLabels = actual_labels;
        
        % % Save results
        % save('results/ecg_classifier_results.mat', 'results_struct');
        % fprintf('\nClassifier and testing results saved to results/ecg_classifier_results.mat\n');
        
        % Visualize confusion matrix
        figure;
        confusionchart(confusion_mat, class_names);
        title('Heartbeat Classification Confusion Matrix');
        
        % % Optional: ROC curve analysis (consider one-vs-all strategy for multi-class cases)
        % if size(prediction_scores, 2) == length(class_names)
        %     fprintf('\nCalculating ROC curves...\n');
        % 
        %     figure;
        %     for i = 1:num_classes
        %         [X, Y, T, AUC] = perfcurve(double(actual_labels == class_names{i}), prediction_scores(:, i), 1);
        %         subplot(1, num_classes, i);
        %         plot(X, Y);
        %         title(sprintf('ROC - %s (AUC: %.3f)', class_names{i}, AUC));
        %         xlabel('False Positive Rate'); ylabel('True Positive Rate');
        %         axis square;
        %         grid on;
        %     end
        % 
        %     saveas(gcf, 'results/roc_curves.fig');
        %     saveas(gcf, 'results/roc_curves.png');
        %     fprintf('ROC curves saved to results folder\n');
        % end
    else
        fprintf('Testing set is empty, unable to evaluate classifier performance.\n');
    end
else
    fprintf('Training feature table is empty, unable to train classifier.\n');
end

fprintf('\nProcessing finished.\n');



