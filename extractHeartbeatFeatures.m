function [featureTable, featureNames] = extractHeartbeatFeatures(beatInfo, fs)
% extractHeartbeatFeatures - Extracts features from heartbeat information structure for machine learning
%
% Input:
%   beatInfo - Struct array, containing heartbeat classification and feature point information (from detectAndClassifyHeartbeats.m)
%     .beatType - Heartbeat type (1:Normal, 5:PVC, 8:PAC)
%     .segment - Extracted ECG segment
%     .rIndex - R-wave index in the segment
%     .pIndex - P-wave index in the segment
%     .qIndex - Q-wave index in the segment
%     .sIndex - S-wave index in the segment
%     .tIndex - T-wave index in the segment
%     .pEnd - P-wave end point index in the segment
%     .tEnd - T-wave end point index in the segment
%     .segmentStartIndex - Starting index of the segment in the original ECG signal
%   fs - Sampling frequency (Hz)
%
% Output:
%   featureTable - Table, containing extracted features and heartbeat type labels
%   featureNames - Cell array, containing feature names

% Define feature names (keep in English for compatibility with machine learning tools)
featureNames = {'RR_Prev', 'RR_Post', 'R_Amplitude', 'P_Amplitude', 'Q_Amplitude', 'S_Amplitude', 'T_Amplitude', ...
                'PR_Interval', 'QRS_Duration', 'ST_Segment', 'QT_Interval', ...
                'P_Duration', 'T_Duration', 'P_Area', 'QRS_Area', 'T_Area', ...
                'R_Peak_Time_From_Prev_R', 'R_Peak_Time_To_Next_R', ... 
                'BeatType'}; % Last column is the label

numBeats = length(beatInfo);
if numBeats == 0
    fprintf('Warning: beatInfo is empty, cannot extract features.\n');
    % Return an empty table with column names consistent with featureNames
    featureTable = array2table(zeros(0, length(featureNames)), 'VariableNames', featureNames);
    % Ensure BeatType column is categorical if possible
    if any(strcmp(featureNames, 'BeatType'))
        featureTable.BeatType = categorical(featureTable.BeatType);
    end
    return;
end

% Initialize feature matrix, each row a heartbeat, each column a feature
% Add 1 because the last column is the BeatType label
features = NaN(numBeats, length(featureNames));

% Calculate indices of all R-waves, used for calculating RR interval
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
    
    % 1. RR interval (RR_Prev, RR_Post)
    % Unit: seconds
    if i > 1
        rr_prev = (allRIndicesInECG(i) - allRIndicesInECG(i-1)) / fs;
        features(i, strcmp(featureNames, 'RR_Prev')) = rr_prev;
        features(i, strcmp(featureNames, 'R_Peak_Time_From_Prev_R')) = rr_prev; % Redundant feature, but sometimes used directly
    else
        features(i, strcmp(featureNames, 'RR_Prev')) = NaN; % First heartbeat has no previous RR
        features(i, strcmp(featureNames, 'R_Peak_Time_From_Prev_R')) = NaN;
    end
    
    if i < numBeats
        rr_post = (allRIndicesInECG(i+1) - allRIndicesInECG(i)) / fs;
        features(i, strcmp(featureNames, 'RR_Post')) = rr_post;
        features(i, strcmp(featureNames, 'R_Peak_Time_To_Next_R')) = rr_post; % Redundant feature
    else
        features(i, strcmp(featureNames, 'RR_Post')) = NaN; % Last heartbeat has no next RR
        features(i, strcmp(featureNames, 'R_Peak_Time_To_Next_R')) = NaN;
    end
    
    % 2. Amplitude features (R, P, Q, S, T)
    % Unit: mV (assuming original ECG unit is mV)
    if ~isnan(rIdx)
        features(i, strcmp(featureNames, 'R_Amplitude')) = segment(rIdx);
    end
    if ~isnan(pIdx)
        features(i, strcmp(featureNames, 'P_Amplitude')) = segment(pIdx);
    end
    if ~isnan(qIdx)
        features(i, strcmp(featureNames, 'Q_Amplitude')) = segment(qIdx);
    end
    if ~isnan(sIdx)
        features(i, strcmp(featureNames, 'S_Amplitude')) = segment(sIdx);
    end
    if ~isnan(tIdx)
        features(i, strcmp(featureNames, 'T_Amplitude')) = segment(tIdx);
    end
    
    % 3. Interval features
    % PR interval (from P-wave start to QRS complex start - usually Q-wave or R-wave)
    % Unit: seconds
    if ~isnan(pIdx) && ~isnan(qIdx) && qIdx > pIdx
        features(i, strcmp(featureNames, 'PR_Interval')) = (qIdx - pIdx) / fs;
    elseif ~isnan(pIdx) && ~isnan(rIdx) && rIdx > pIdx % If no Q-wave, use R-wave instead
        features(i, strcmp(featureNames, 'PR_Interval')) = (rIdx - pIdx) / fs;
    end
    
    % QRS duration (from Q-wave to S-wave end)
    % Unit: seconds
    if ~isnan(qIdx) && ~isnan(sIdx) && sIdx > qIdx
        features(i, strcmp(featureNames, 'QRS_Duration')) = (sIdx - qIdx) / fs;
    elseif ~isnan(rIdx) && ~isnan(sIdx) && sIdx > rIdx % If no Q-wave, calculate from R-wave to S-wave
        features(i, strcmp(featureNames, 'QRS_Duration')) = (sIdx - rIdx) / fs;
    elseif ~isnan(qIdx) && ~isnan(rIdx) && rIdx > qIdx % If no S-wave, calculate from Q-wave to R-wave
        features(i, strcmp(featureNames, 'QRS_Duration')) = (rIdx - qIdx) / fs;
    end
    
    % ST segment (from S-wave end to T-wave start - simplified here as time from S-wave to T-wave peak)
    % Unit: seconds
    if ~isnan(sIdx) && ~isnan(tIdx) && tIdx > sIdx
        features(i, strcmp(featureNames, 'ST_Segment')) = (tIdx - sIdx) / fs;
    end
    
    % QT interval (from Q-wave start to T-wave end)
    % Unit: seconds
    if ~isnan(qIdx) && ~isnan(tEndIdx) && tEndIdx > qIdx
        features(i, strcmp(featureNames, 'QT_Interval')) = (tEndIdx - qIdx) / fs;
    elseif ~isnan(rIdx) && ~isnan(tEndIdx) && tEndIdx > rIdx % If no Q-wave, use R-wave
         features(i, strcmp(featureNames, 'QT_Interval')) = (tEndIdx - rIdx) / fs;
    end
    
    % P-wave duration (from P-wave start to P-wave end pEndIdx)
    % Unit: seconds
    if ~isnan(pIdx) && ~isnan(pEndIdx) && pEndIdx > pIdx
        features(i, strcmp(featureNames, 'P_Duration')) = (pEndIdx - pIdx) / fs;
    end

    % T-wave duration (from T-wave start to T-wave end tEndIdx)
    % Unit: seconds
    % Assume T-wave start can be estimated by (tIdx - 0.05*fs) if T-wave onset is not precisely detected
    % Or, if tIdx is the peak, can be defined from a point to tEndIdx
    % Simplified here: if tIdx is known, T-wave duration from a point before tIdx (rough start) to tEndIdx
    if ~isnan(tIdx) && ~isnan(tEndIdx) && tEndIdx > tIdx
        t_start_approx = max(1, tIdx - round(0.05*fs)); % Rough T-wave start
        if tEndIdx > t_start_approx
            features(i, strcmp(featureNames, 'T_Duration')) = (tEndIdx - t_start_approx) / fs;
        end
    end
    
    % 4. Area features (approximate calculation, e.g., sum under trapezoidal rule)
    % P-wave area (from pIdx to pEndIdx)
    if ~isnan(pIdx) && ~isnan(pEndIdx) && pEndIdx > pIdx && pEndIdx <= length(segment)
        p_wave_segment = segment(pIdx:pEndIdx);
        features(i, strcmp(featureNames, 'P_Area')) = sum(p_wave_segment) / fs; % Approximate integral
    end
    
    % QRS complex area (from qIdx to sIdx)
    if ~isnan(qIdx) && ~isnan(sIdx) && sIdx > qIdx && sIdx <= length(segment)
        qrs_segment = segment(qIdx:sIdx);
        features(i, strcmp(featureNames, 'QRS_Area')) = sum(qrs_segment) / fs; % Approximate integral
    elseif ~isnan(rIdx) % If Q or S is missing, simplify to a small window around R-wave
        r_area_start = max(1, rIdx - round(0.02*fs));
        r_area_end = min(length(segment), rIdx + round(0.02*fs));
        if r_area_end > r_area_start
            qrs_segment = segment(r_area_start:r_area_end);
            features(i, strcmp(featureNames, 'QRS_Area')) = sum(qrs_segment) / fs;
        end
    end
    
    % T-wave area (from some T-wave start to tEndIdx)
    if ~isnan(tIdx) && ~isnan(tEndIdx) && tEndIdx > tIdx && tEndIdx <= length(segment)
        t_start_approx = max(1, tIdx - round(0.05*fs)); % Rough T-wave start
        if tEndIdx > t_start_approx
            t_wave_segment = segment(t_start_approx:tEndIdx);
            features(i, strcmp(featureNames, 'T_Area')) = sum(t_wave_segment) / fs; % Approximate integral
        end
    end
    
    % 5. Heartbeat type label
    features(i, strcmp(featureNames, 'BeatType')) = currentBeat.beatType;
end

% Convert feature matrix to table
featureTable = array2table(features, 'VariableNames', featureNames);

% Ensure BeatType column is categorical
% This may be important for some MATLAB classification tools
if any(strcmp(featureNames, 'BeatType')) && ~isempty(featureTable)
    try
        featureTable.BeatType = categorical(featureTable.BeatType);
    catch ME_cat
        fprintf('Warning: Unable to convert BeatType column to categorical type: %s\n', ME_cat.message);
        fprintf('Please check if the BeatType column contains non-numeric or unsuitable values for categorization.\n');
    end
end


% Can add some feature statistics or visualization (if needed)
% For example: disp(summary(featureTable));

fprintf('Feature extraction completed, extracted %d features for %d heartbeats.\n', numBeats, length(featureNames)-1); % -1 because BeatType is a label

end

