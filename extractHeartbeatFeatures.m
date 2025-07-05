function [featureTable, featureNames] = extractHeartbeatFeatures(beatInfo, fs)
% extractHeartbeatFeatures - Extracts features from heartbeat info struct for machine learning
%
% Inputs:
%   beatInfo - Struct array containing heartbeat classification and feature point information (from detectAndClassifyHeartbeats.m)
%     .beatType - Heartbeat type ('Other': Other, 'Normal': Normal, 'PVC': PVC, 'PAC': PAC)
%     .segment - Extracted ECG segment
%     .rIndex - Index of the R-wave within the segment
%     .pIndex - Index of the P-wave within the segment
%     .qIndex - Index of the Q-wave within the segment
%     .sIndex - Index of the S-wave within the segment
%     .tIndex - Index of the T-wave within the segment
%     .pEnd - End point of the P-wave within the segment
%     .tEnd - End point of the T-wave within the segment
%     .segmentStartIndex - Starting index of the segment in the original ECG signal
%   fs - Sampling frequency (Hz)
%
% Outputs:
%   featureTable - Table, containing extracted features and heartbeat type labels (normalized)
%   featureNames - Cell array, containing feature names

% Feature names definition (removed R_Peak_Time_From_Prev_R, R_Peak_Time_To_Next_R, ST_Segment)
% 
% *** Core Feature Selection Explanation ***
% Based on medical knowledge and feature importance analysis, the most important features for PVC vs. Other binary classification are:
% 1. RR_Prev, RR_Post - Key features for premature beats (rhythm changes)
% 2. QRS_Duration - Core feature for PVC (ventricular premature beats usually have a wider QRS)
% 3. R_Amplitude - Main wave amplitude feature
%
% Currently extracting all features, using 4 core features during training to avoid overfitting
featureNames = {'RR_Prev', 'RR_Post', 'R_Amplitude', 'Q_Amplitude', 'S_Amplitude', 'T_Amplitude', ...
                'QRS_Duration', 'QT_Interval', ...
                'T_Duration', 'QRS_Area', 'T_Area', ...
                'BeatType'}; % The last column is the label

numBeats = length(beatInfo);
if numBeats == 0
    fprintf('Warning: beatInfo is empty, cannot extract features.\n');
    % Return an empty table with column names matching featureNames
    featureTable = array2table(zeros(0, length(featureNames)), 'VariableNames', featureNames);
    % Ensure BeatType column is categorical if possible
    if any(strcmp(featureNames, 'BeatType'))
        featureTable.BeatType = categorical(featureTable.BeatType);
    end
    return;
end

% Initialize feature matrix and label array
% Numerical feature matrix (excluding BeatType)
numericalFeatureNames = setdiff(featureNames, {'BeatType'});
features = NaN(numBeats, length(numericalFeatureNames));

% Process labels separately
beatTypeLabels = cell(numBeats, 1);

% Calculate indices of all R-waves to compute RR intervals
allRIndicesInECG = zeros(numBeats, 1);
for k = 1:numBeats
    % beatInfo(k).rIndex is the index within the segment
    % beatInfo(k).segmentStartIndex is the start index of the segment in the original ECG
    allRIndicesInECG(k) = beatInfo(k).segmentStartIndex + beatInfo(k).rIndex - 1;
end

for i = 1:numBeats
    currentBeat = beatInfo(i);
    segment = currentBeat.segment;
    rIdx = currentBeat.rIndex;
    pIdx = currentBeat.pIndex;
    qIdx = currentBeat.qIndex;
    sIdx = currentBeat.sIndex;
    tIdx = currentBeat.tIndex;
    pEndIdx = currentBeat.pEnd;
    tEndIdx = currentBeat.tEnd;
    
    % 1. RR interval features (RR_Prev, RR_Post)
    % Unit: seconds
    if i > 1
        rr_prev = (allRIndicesInECG(i) - allRIndicesInECG(i-1)) / fs;
        features(i, strcmp(numericalFeatureNames, 'RR_Prev')) = rr_prev;
    else
        features(i, strcmp(numericalFeatureNames, 'RR_Prev')) = NaN; % The first heartbeat has no preceding RR
    end
    
    if i < numBeats
        rr_post = (allRIndicesInECG(i+1) - allRIndicesInECG(i)) / fs;
        features(i, strcmp(numericalFeatureNames, 'RR_Post')) = rr_post;
    else
        features(i, strcmp(numericalFeatureNames, 'RR_Post')) = NaN; % The last heartbeat has no subsequent RR
    end
    
    % 2. Amplitude features (R, Q, S, T) - P_Amplitude removed
    % Unit: mV (assuming the original ECG unit is mV)
    if ~isnan(rIdx) && rIdx > 0 && rIdx <= length(segment)
        features(i, strcmp(numericalFeatureNames, 'R_Amplitude')) = segment(rIdx);
    end
    if ~isnan(qIdx) && qIdx > 0 && qIdx <= length(segment)
        features(i, strcmp(numericalFeatureNames, 'Q_Amplitude')) = segment(qIdx);
    end
    if ~isnan(sIdx) && sIdx > 0 && sIdx <= length(segment)
        features(i, strcmp(numericalFeatureNames, 'S_Amplitude')) = segment(sIdx);
    end
    if ~isnan(tIdx) && tIdx > 0 && tIdx <= length(segment)
        features(i, strcmp(numericalFeatureNames, 'T_Amplitude')) = segment(tIdx);
    end
    
    % 3. Interval features - PR_Interval removed
    % QRS duration (corrected calculation method)
    % Unit: seconds
    if ~isnan(qIdx) && ~isnan(sIdx) && sIdx > qIdx
        features(i, strcmp(numericalFeatureNames, 'QRS_Duration')) = (sIdx - qIdx) / fs;
    elseif ~isnan(rIdx) && ~isnan(sIdx) && sIdx > rIdx % If no Q-wave, calculate from R to S
        features(i, strcmp(numericalFeatureNames, 'QRS_Duration')) = (sIdx - rIdx) / fs;
    elseif ~isnan(qIdx) && ~isnan(rIdx) && rIdx > qIdx % If no S-wave, calculate from Q to R
        features(i, strcmp(numericalFeatureNames, 'QRS_Duration')) = (rIdx - qIdx) / fs;
    else
        % If both Q and S are not detected, use a fixed QRS width estimate (approx. 0.08s)
        features(i, strcmp(numericalFeatureNames, 'QRS_Duration')) = 0.08;
    end
    
    % QT interval (from Q-wave start to T-wave end)
    % Unit: seconds
    if ~isnan(qIdx) && ~isnan(tEndIdx) && tEndIdx > qIdx
        features(i, strcmp(numericalFeatureNames, 'QT_Interval')) = (tEndIdx - qIdx) / fs;
    elseif ~isnan(rIdx) && ~isnan(tEndIdx) && tEndIdx > rIdx % If no Q-wave, use R-wave
        features(i, strcmp(numericalFeatureNames, 'QT_Interval')) = (tEndIdx - rIdx) / fs;
    end
    
    % T-wave duration (corrected calculation method)
    % Unit: seconds
    if ~isnan(tIdx) && ~isnan(tEndIdx) && tEndIdx > tIdx
        % Use a more accurate T-wave start estimation method
        t_start_approx = max(1, tIdx - round(0.08*fs)); % T-wave start is usually 80ms before the peak
        if tEndIdx > t_start_approx && t_start_approx <= length(segment)
            features(i, strcmp(numericalFeatureNames, 'T_Duration')) = (tEndIdx - t_start_approx) / fs;
        end
    end
    
    % 4. Area features (corrected calculation method) - P_Area removed
    % QRS complex area (using trapezoidal integration)
    if ~isnan(qIdx) && ~isnan(sIdx) && sIdx > qIdx && sIdx <= length(segment)
        qrs_segment = segment(qIdx:sIdx);
        features(i, strcmp(numericalFeatureNames, 'QRS_Area')) = trapz(abs(qrs_segment)) / fs; % Use absolute value
    elseif ~isnan(rIdx) % If Q or S is missing, use a window around the R-wave
        r_area_start = max(1, rIdx - round(0.04*fs)); % Widen window to ±40ms
        r_area_end = min(length(segment), rIdx + round(0.04*fs));
        if r_area_end > r_area_start
            qrs_segment = segment(r_area_start:r_area_end);
            features(i, strcmp(numericalFeatureNames, 'QRS_Area')) = trapz(abs(qrs_segment)) / fs;
        end
    end
    
    % T-wave area (using trapezoidal integration)
    if ~isnan(tIdx) && ~isnan(tEndIdx) && tEndIdx > tIdx && tEndIdx <= length(segment)
        t_start_approx = max(1, tIdx - round(0.08*fs)); % Use the same start point as T_Duration
        if tEndIdx > t_start_approx && t_start_approx <= length(segment)
            t_wave_segment = segment(t_start_approx:tEndIdx);
            features(i, strcmp(numericalFeatureNames, 'T_Area')) = trapz(t_wave_segment) / fs;
        end
    end
    
    % 5. Heartbeat type label (processed separately)
    if isa(currentBeat.beatType, 'char') || isa(currentBeat.beatType, 'string')
        beatTypeLabels{i} = char(currentBeat.beatType); % Ensure it is a char type
    else
        beatTypeLabels{i} = 'Other'; % Default value
    end
end

% First create the table for numerical features
numericalFeatureTable = array2table(features, 'VariableNames', numericalFeatureNames);

% Add the BeatType column
numericalFeatureTable.BeatType = beatTypeLabels;

% Reorder columns to ensure BeatType is last
featureTable = numericalFeatureTable(:, featureNames);

% Feature normalization (excluding BeatType column)
fprintf('Starting feature normalization...\n');
featureColumns = setdiff(featureNames, {'BeatType'}); % Exclude BeatType column

for i = 1:length(featureColumns)
    colName = featureColumns{i};
    colData = featureTable{:, colName};
    
    % Remove NaN values for statistics
    validData = colData(~isnan(colData));
    
    if ~isempty(validData) && length(validData) > 1
        % Calculate mean and standard deviation
        meanVal = mean(validData);
        stdVal = std(validData);
        
        % Z-score normalization (if standard deviation is not 0)
        if stdVal > 1e-10 % Avoid division by zero
            normalizedData = (colData - meanVal) / stdVal;
            featureTable{:, colName} = normalizedData;
            fprintf('Feature %s: Mean=%.4f, Std Dev=%.4f, normalized\n', colName, meanVal, stdVal);
        else
            % If standard deviation is 0, perform min-max normalization
            minVal = min(validData);
            maxVal = max(validData);
            if maxVal > minVal
                normalizedData = (colData - minVal) / (maxVal - minVal);
                featureTable{:, colName} = normalizedData;
                fprintf('Feature %s: Std Dev is 0, using min-max normalization [%.4f, %.4f]\n', colName, minVal, maxVal);
            else
                fprintf('Feature %s: All values are the same, skipping normalization\n', colName);
            end
        end
    else
        fprintf('Feature %s: Not enough valid data, skipping normalization\n', colName);
    end
end

% Ensure the BeatType column is of string type
if any(strcmp(featureNames, 'BeatType')) && ~isempty(featureTable)
    % BeatType column is already a cell array, convert to string array
    if iscell(featureTable.BeatType)
        featureTable.BeatType = string(featureTable.BeatType);
    end
    fprintf('BeatType column has been ensured to be of string type\n');
end

% Output feature extraction statistics
fprintf('Feature extraction complete, extracted %d features for %d heartbeats (normalized).\n', length(featureNames)-1, numBeats);

% Display the number of each beat type
if ~isempty(featureTable)
    beatTypes = featureTable.BeatType;
    uniqueTypes = unique(beatTypes);
    fprintf('Heartbeat type distribution:\n');
    for i = 1:length(uniqueTypes)
        count = sum(strcmp(beatTypes, uniqueTypes{i}));
        fprintf('  %s: %d heartbeats (%.1f%%)\n', uniqueTypes{i}, count, (count/numBeats)*100);
    end
end

% Output core feature usage recommendation
fprintf('\n*** Feature Selection Recommendation ***\n');
fprintf('To improve PVC recognition rate and avoid overfitting, it is recommended to use only the following core features when training the classifier:\n');
fprintf('  1. RR_Prev, RR_Post - Rhythm change features for premature beats\n');
fprintf('  2. QRS_Duration - Key feature for PVC (wide QRS)\n');
fprintf('  3. R_Amplitude - Main wave amplitude feature\n');
fprintf('trainClassifier.m is currently configured to use these 4 core features for PVC vs Other binary classification.\n');
fprintf('Other features have been extracted but are commented out during training. To enable them, please modify predictorNames in trainClassifier.m.\n\n');

end

