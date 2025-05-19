% load_mit_incart_data.m
% Description: This script uses the WFDB MATLAB toolkit to load all ECG records and their standard annotations from the local '../data/' path for MIT and INCART databases,
% and saves the data for each database separately into .mat files in the results directory.
%
% Usage:
% 1. Ensure the WFDB MATLAB toolbox is installed and in the MATLAB path.
% 2. Ensure the '../data/MIT/' and '../data/INCART/' directory structures exist and contain the respective database files.
% 3. Run this script directly in MATLAB. The output .mat files will be saved in the 'results' folder.
clc;
clear;

fprintf('Starting ECG data loading and saving script...\n\n');

% Save current path to restore later
original_path = pwd;
fprintf('Current working directory: %s\n', original_path);

% Define and create results directory
results_path = fullfile(original_path, 'results');
if ~isfolder(results_path)
    try
        mkdir(results_path);
        fprintf('Created results directory: %s\n', results_path);
    catch ME_mkdir
        fprintf('Error: Could not create results directory %s: %s\n', results_path, ME_mkdir.message);
        fprintf('Please check permissions or create the directory manually.');
        return; % Exit script if directory cannot be created
    end
end
fprintf('Data will be saved to: %s\n', results_path);


% Define data root directory (revised to relative path)
data_root_path = '/data/'; 
fprintf('Data source root directory: %s (relative to %s)\n', data_root_path, original_path);

% Define database names
% databases = {'MIT', 'INCART', 'european-st-t-database-1.0.0'};
% databases = {'sudden-cardiac-death-holter-database-1.0.0'};
% databases = {'mit-bih-long-term-ecg-database-1.0.0', ...
%     'leipzig-heart-center-ecg-database-arrhythmias-in-children-and-patients-with-congenital-heart-disease-1.0.0'};
databases = {'shdb-af-a-japanese-holter-ecg-database-of-atrial-fibrillation-1.0.1'};

% Check if WFDB toolbox functions are available (simple check for rdsamp and rdann)
if ~exist('rdsamp', 'file') 
    error('rdsamp function from WFDB MATLAB toolbox not found. Please ensure it is correctly installed and added to the MATLAB path.');
end
if ~exist('rdann', 'file')
    error('rdann function from WFDB MATLAB toolbox not found. Please ensure it is correctly installed and added to the MATLAB path.');
end
fprintf('WFDB tool functions check passed.\n\n');

% Loop through each database
for db_idx = 1:length(databases)
    current_database_name = databases{db_idx};
    % Construct full path to the specific database
    database_full_path = fullfile(original_path, data_root_path, current_database_name);
    
    fprintf('------------------------------------------------------------\n');
    fprintf('Starting processing for database: %s\n', current_database_name);
    fprintf('Target database path: %s\n', database_full_path);

    % Check if database path exists
    if ~isfolder(database_full_path)
        fprintf('Error: Database path %s does not exist. Skipping this database.\n\n', database_full_path);
        continue;
    end

    % Try to change to database directory
    try
        cd(database_full_path);
        fprintf('Successfully changed working directory to: %s\n', pwd);
    catch ME_cd
        fprintf('Error: Cannot change directory to %s: %s.\n', database_full_path, ME_cd.message);
        fprintf('Please check if the path is correct and if you have access permissions. Skipping this database.\n\n');
        cd(original_path); % Change back to original path
        continue;
    end

    % Get all records in the database (based on .hea header files)
    header_files = dir('*.hea');
    
    if isempty(header_files)
        fprintf('Warning: No .hea files found in %s (%s). This database might be empty or not contain records.\n', current_database_name, pwd);
        cd(original_path); % Change back to original path
        fprintf('\n');
        continue;
    end

    fprintf('Found %d records in %s database (based on .hea files).\n', length(header_files), current_database_name);

    % Initialize struct array to store all record data for the current database
    all_records_data_for_db = []; 

    % Loop through each record
    for rec_idx = 1:length(header_files)
        record_header_filename = header_files(rec_idx).name;
        record_name = record_header_filename(1:end-4); 

        fprintf('\n  --- Starting to load record: %s ---\n', record_name);
        
        % Initialize data structure for the current record
        current_record_struct = struct();
        current_record_struct.name = record_name;
        current_record_struct.fs = NaN; % Default value
        current_record_struct.tm = [];
        current_record_struct.signal_data = [];
        current_record_struct.annotations = [];
        current_record_struct.anntype = '';
        current_record_struct.subtype = [];
        current_record_struct.chan = [];
        current_record_struct.num = [];
        current_record_struct.comments = {};

        try
            % Load signal data using rdsamp
            [signal_data, fs, tm] = rdsamp(record_name);
            
            if isempty(signal_data)
                fprintf('    Warning: rdsamp returned empty signal data for record %s. File might be corrupted or format not supported.\n', record_name);
            else
                fprintf('    Signal data (%s) loaded successfully:\n', record_name);
                fprintf('      - Sampling rate (fs): %d Hz\n', fs);
                if ~isempty(tm)
                    fprintf('      - Record duration: %.2f seconds (based on time vector tm)\n', tm(end));
                else
                    fprintf('      - Time vector tm is empty.\n');
                end
                fprintf('      - Signal dimensions: %d (samples) x %d (channels)\n', size(signal_data, 1), size(signal_data, 2));
                
                current_record_struct.fs = fs;
                current_record_struct.tm = tm;
                current_record_struct.signal_data = signal_data;
            end

            % Try to load annotation data
            atr_file_full_path = fullfile(pwd, [record_name '.atr']);
            if isfile(atr_file_full_path)
                [annotations, anntype, subtype, chan, num, comments_ann] = rdann(record_name, 'atr'); % Renamed comments to avoid conflict
                 if isempty(annotations) && isempty(anntype)
                    fprintf('    Annotation data (%s.atr): File exists but no annotations loaded, or annotations are empty.\n', record_name);
                else
                    fprintf('    Annotation data (%s.atr) loaded successfully:\n', record_name);
                    fprintf('      - Found %d annotation markers.\n', length(annotations));
                    current_record_struct.annotations = annotations;
                    current_record_struct.anntype = anntype;
                    current_record_struct.subtype = subtype;
                    current_record_struct.chan = chan;
                    current_record_struct.num = num;
                    current_record_struct.comments = comments_ann;
                 end
            else
                fprintf('    Standard annotation file (%s.atr) not found for record %s.\n', record_name, record_name);
            end
            
            fprintf('    Data loading for record %s completed, preparing to store.\n', record_name);
            % Append current record's struct to the total data for the database
            if isempty(all_records_data_for_db)
                all_records_data_for_db = current_record_struct;
            else
                all_records_data_for_db(end+1) = current_record_struct;
            end

        catch ME_load_record
            fprintf('    Error occurred while loading record %s: %s\n', record_name, ME_load_record.message);
            fprintf('    Error details (identifier: %s):\n', ME_load_record.identifier);
            for k_stack = 1:length(ME_load_record.stack)
                fprintf('      In %s (file: %s, line: %d)\n', ME_load_record.stack(k_stack).name, ME_load_record.stack(k_stack).file, ME_load_record.stack(k_stack).line);
            end
            fprintf('    Record %s could not be fully loaded and may not be saved or saved with incomplete data.\n', record_name);
            % Even if an error occurs, try to save partially collected information (if rdsamp succeeded but rdann failed)
            if ~isempty(current_record_struct.name) % Ensure at least there is a name
                if isempty(all_records_data_for_db)
                    all_records_data_for_db = current_record_struct;
                else
                    all_records_data_for_db(end+1) = current_record_struct; 
                end
            end
        end
        fprintf('  --- Finished processing record: %s ---\n', record_name);
    end % End record loop
    
    % After processing all records in a database, save data to .mat file
    if ~isempty(all_records_data_for_db)
        mat_filename = fullfile(results_path, [current_database_name '_data.mat']);
        fprintf('\nPreparing to save all loaded record data for database %s to file: %s\n', current_database_name, mat_filename);
        try
            variable_name_for_mat_file = matlab.lang.makeValidName([current_database_name '_data']);
            temp_save_struct = struct();
            temp_save_struct.(variable_name_for_mat_file) = all_records_data_for_db;
            save(mat_filename, '-struct', 'temp_save_struct');
            fprintf('Data for database %s successfully saved to %s (as variable \'\'%s\'\')\n', current_database_name, mat_filename, variable_name_for_mat_file);
        catch ME_save
            fprintf('Error: Failed to save data for database %s to %s: %s\n', current_database_name, mat_filename, ME_save.message);
        end
    else
        fprintf('\nNo valid record data loaded for database %s, .mat file will not be created.\n', current_database_name);
    end

    % Change back to original path
    cd(original_path);
    fprintf('\nSwitched back from %s to original working directory: %s\n', current_database_name, pwd);
    fprintf('Finished processing database: %s\n', current_database_name);
    fprintf('------------------------------------------------------------\n\n');
end % End database loop

fprintf('All database processing completed.\n');
fprintf('Script execution finished.\n');