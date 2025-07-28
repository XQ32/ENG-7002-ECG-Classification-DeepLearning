
%% Main Function: ECG Heartbeat Detection, Classification and Deep Learning Model Training (Enhanced Version)
%
% Function Description:
% This script is the main controller for the entire ECG analysis pipeline. It coordinates the following key steps:
% 1. Set configuration parameters, including database paths and allocations.
% 2. Load ECG data and annotations from .mat files.
% 3. Filter raw ECG signals and perform heartbeat detection.
% 4. Clean data, remove invalid heartbeat records.
% 5. Count and display the number of each type of heartbeat.
% 6. Prepare, balance and save data for deep learning models.
% 7. Build, train and evaluate deep learning models on test sets (supports multiple network architectures).
%
% Script Structure:
% - Each major step is encapsulated in independent functions to improve code modularity and readability.
% - The main function is responsible for calling these modules in sequence and passing necessary parameters.
% - Supports two architecture choices: simple CNN and multi-modal attention network.

clc;
clear;
close all;

%% 1. Initialization and Configuration
fprintf('--- Step 1: Load Configuration ---\n');
[db_allocation, mat_files_info, results_dir, dl_params] = setupConfiguration();

%% Check Data Cache
[useCache, allBeatInfo, training_waveforms, training_context, training_labels, testing_waveforms, testing_context, testing_labels] = ...
    checkAndLoadCachedData(mat_files_info, db_allocation, dl_params);

if ~useCache
    %% 2. Load and Process ECG Data
    fprintf('--- Step 2: Load and Process ECG Data ---\n');
    allBeatInfo = loadAndProcessECGData(mat_files_info, results_dir);
    fprintf('All database processing completed.\n\n');

    %% 3. Data Cleaning
    fprintf('--- Step 3: Clean Heartbeat Data ---\n');
    allBeatInfo = cleanHeartbeatData(allBeatInfo);
    fprintf('Data cleaning completed.\n\n');

    %% 4. Statistical Analysis
    fprintf('--- Step 4: Display Heartbeat Statistics ---\n');
    displayBeatStatistics(allBeatInfo);
    fprintf('Statistical analysis completed.\n\n');

    %% 5. Prepare Deep Learning Data
    fprintf('--- Step 5: Prepare Deep Learning Data ---\n');
    [training_waveforms, training_context, training_labels, testing_waveforms, testing_context, testing_labels] = ...
        createDLDatasets(allBeatInfo, db_allocation, dl_params);
    fprintf('Deep learning data preparation completed.\n\n');

    %% Save Data Cache
    saveCachedData(allBeatInfo, training_waveforms, training_context, training_labels, ...
                   testing_waveforms, testing_context, testing_labels, ...
                   mat_files_info, db_allocation, dl_params);
else
    fprintf('Using cached data, skipping steps 2-5\n\n');
end

%% 6. Train and Evaluate Model
fprintf('--- Step 6: Train and Evaluate Deep Learning Model ---\n');
trainAndEvaluateModel(training_waveforms, training_context, training_labels, ...
                      testing_waveforms, testing_context, testing_labels, dl_params);
fprintf('Model training and evaluation completed.\n\n');

fprintf('=== Entire pipeline execution completed ===\n');





