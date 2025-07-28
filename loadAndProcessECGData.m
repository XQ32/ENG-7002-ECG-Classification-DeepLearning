function allBeatInfo = loadAndProcessECGData(mat_files_info, results_dir)
%loadAndProcessECGData Load ECG data from .mat files, process and extract heartbeat information
%   This function iterates through all specified database files, loads signals and annotations,
%   then performs filtering and heartbeat detection, finally returns a structure array containing all heartbeat information.
%
%   Input:
%       mat_files_info  - Cell array containing detailed information for each database file
%       results_dir     - String, directory storing .mat files
%
%   Output:
%       allBeatInfo     - Structure array containing detailed information of all detected heartbeats

%% Initialize Variables
initial_struct = struct('beatType', 'Other', 'segment', [], 'rIndex', 0, 'pIndex', NaN, ...
                'qIndex', NaN, 'sIndex', NaN, 'tIndex', NaN, 'pEnd', NaN, 'tEnd', NaN, ...
                'segmentStartIndex', 0, 'fs', 0, 'originalRecordName', '', ...
                'originalDatabaseName', '', 'originalAnnotationChar', '');

chunk_size = 100000;
allBeatInfo = repmat(initial_struct, chunk_size, 1);
actual_beat_count = 0;
allocated_size = chunk_size;

fprintf('Using dynamic memory allocation strategy, initially allocated %d rows, automatically expanding as needed\n', chunk_size);

%% Process Data Records Individually
fprintf('Starting to process database .mat files, detecting heartbeats and extracting features...\n');

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
        fprintf('Warning: Expected structure variable ''%s'' not found in %s. Skipping this database.\n', current_db_info.structName, current_db_info.fileName);
        clear loaded_data;
        continue;
    end

    db_records_array = loaded_data.(current_db_info.structName);
    fprintf('Database %s contains %d records.\n', current_db_info.structName, length(db_records_array));
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
        M = current_record.signal_data(:, 1);

        if isempty(current_record.annotations) || isempty(current_record.anntype)
             fprintf('    Warning: Annotations for record %s are empty or type is empty. Skipping this record.\n', recordName);
            continue;
        end

        ATRTIMED = double(current_record.annotations) / fs;
        ANNOT_CHAR_orig = current_record.anntype;

        ANNOTD = cell(size(ANNOT_CHAR_orig, 1), 1);
        for k_ann = 1:size(ANNOT_CHAR_orig, 1)
            ann_char_single = strtrim(ANNOT_CHAR_orig(k_ann,:));
            switch upper(ann_char_single)
                case 'V'
                    ANNOTD{k_ann} = 'PVC';
                otherwise
                    ANNOTD{k_ann} = 'Other';
            end
        end

        try
            power_line_freq = current_db_info.powerLineFreq;
            fprintf('    Using filter to remove %d Hz power line interference\n', power_line_freq);
            [ecg_filtered, ~] = ecgFilter(M, fs, 2, power_line_freq);

            [~, beatInfo_rec_numeric] = detectAndClassifyHeartbeats(ecg_filtered, ATRTIMED, ANNOTD, fs);

            if isempty(beatInfo_rec_numeric)
                fprintf('  Warning: Heartbeat detection result for record %s is empty, skipping this record.\n', recordName);
                clear M ecg_filtered ATRTIMED ANNOT_CHAR_orig ANNOTD;
                continue;
            end

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
                    tempBeatInfoArray(j).fs = fs;
                    tempBeatInfoArray(j).originalRecordName = recordName;
                    tempBeatInfoArray(j).originalDatabaseName = current_db_info.structName;

                    r_peak_in_original_ecg = beatInfo_rec_numeric(j).segmentStartIndex + beatInfo_rec_numeric(j).rIndex - 1;
                    [~, closest_idx_in_orig_ann] = min(abs(double(current_record.annotations) - r_peak_in_original_ecg));
                    
                    if abs(double(current_record.annotations(closest_idx_in_orig_ann)) - r_peak_in_original_ecg) <= 2
                        tempBeatInfoArray(j).originalAnnotationChar = strtrim(ANNOT_CHAR_orig(closest_idx_in_orig_ann,:));
                    else
                        tempBeatInfoArray(j).originalAnnotationChar = '?';
                    end
                end
                beatInfo_rec = tempBeatInfoArray;
                
                clear tempBeatInfoArray beatInfo_rec_numeric;
            else
                beatInfo_rec = [];
            end

            new_beats_count = length(beatInfo_rec);
            if actual_beat_count + new_beats_count > allocated_size
                additional_chunks = ceil((actual_beat_count + new_beats_count - allocated_size) / chunk_size);
                new_allocation = allocated_size + additional_chunks * chunk_size;
                fprintf('    Expanding memory: from %d rows to %d rows\n', allocated_size, new_allocation);
                
                temp_allBeatInfo = repmat(initial_struct, new_allocation, 1);
                temp_allBeatInfo(1:actual_beat_count) = allBeatInfo(1:actual_beat_count);
                clear allBeatInfo;
                allBeatInfo = temp_allBeatInfo;
                allocated_size = new_allocation;
                clear temp_allBeatInfo;
            end
            
            allBeatInfo(actual_beat_count + 1:actual_beat_count + new_beats_count) = beatInfo_rec;
            actual_beat_count = actual_beat_count + new_beats_count;
    
            clear M ecg_filtered beatInfo_rec ATRTIMED ANNOT_CHAR_orig ANNOTD;
            
        catch ME_processing
            fprintf('  Error occurred while processing record %s: %s\n  Skipping this record and continuing processing.\n', recordName, ME_processing.message);
            clear M ecg_filtered ATRTIMED ANNOT_CHAR_orig ANNOTD;
            continue;
        end
    end

    clear db_records_array current_db_info;
end

fprintf('\n=== Data processing completed, starting memory optimization ===\n');
fprintf('Actually used: %d rows, allocated: %d rows, memory utilization: %.1f%%\n', ...
    actual_beat_count, allocated_size, (actual_beat_count/allocated_size)*100);

if actual_beat_count < allocated_size
    allBeatInfo = allBeatInfo(1:actual_beat_count);
    fprintf('Memory shrunk to %d rows\n', actual_beat_count);
end

end