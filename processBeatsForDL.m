function [X_waveforms, X_context, Y_labels] = processBeatsForDL(beatInfo, fs, params)
% processBeatsForDL - Prepares and preprocesses data for deep learning models (enhanced version)
%
% This function prepares data for multi-modal attention networks, supporting:
% 1. High-quality waveform data preprocessing and multi-scale normalization
% 2. Enhanced context feature extraction (RR intervals + heart rate variability metrics)
% 3. Intelligent data quality detection and filtering
% 4. Reserved interface for subsequent multi-scale morphological feature extraction
%
% Input:
%   beatInfo - Structure array containing heartbeat information (from main_v3.m)
%   fs - Sampling frequency of all heartbeats in beatInfo (Hz)
%   params   - Structure containing preprocessing parameters
%     .standardLength - Standard length to which all heartbeat waveforms should be resampled
%     .enableQualityCheck - (Optional) Whether to enable data quality detection, default true
%     .enhancedContext - (Optional) Whether to enable enhanced context features, default true
%     .normalizationMethod - (Optional) Normalization method: 'zscore'(default) | 'minmax' | 'robust'
%
% Output:
%   X_waveforms - N x 1 cell array, each cell is a 1 x L standardized waveform
%   X_context   - N x C matrix, containing standardized context features (C=2 or more, depending on configuration)
%   Y_labels    - N x 1 categorical label array

% Parse parameters, set default values
if ~isfield(params, 'enableQualityCheck'), params.enableQualityCheck = true; end
if ~isfield(params, 'enhancedContext'), params.enhancedContext = true; end
if ~isfield(params, 'normalizationMethod'), params.normalizationMethod = 'zscore'; end

standardLength = params.standardLength;

% Check if beatInfo is empty
numBeats = length(beatInfo);
if numBeats == 0
    fprintf('Warning: beatInfo is empty, cannot prepare deep learning data.\n');
    X_waveforms = {};
    X_context = [];
    Y_labels = categorical([]);
    return;
end

% Initialize output variables
X_waveforms = cell(numBeats, 1);
Y_labels_cell = cell(numBeats, 1);

% Determine context feature dimension based on whether enhanced context is enabled
if params.enhancedContext
    % Enhanced mode: RR_Prev, RR_Post, HR_Variability, Rhythm_Stability
    contextDim = 4;
    contextNames = {'RR_Prev', 'RR_Post', 'HR_Variability', 'Rhythm_Stability'};
else
    % Basic mode: RR_Prev, RR_Post (for backward compatibility)
    contextDim = 2;
    contextNames = {'RR_Prev', 'RR_Post'};
end

X_context = NaN(numBeats, contextDim);

% --- Step 1: Extract basic RR interval features ---
fprintf('    Extracting RR interval features...\n');
allRIndicesInECG = zeros(numBeats, 1);
for k = 1:numBeats
    allRIndicesInECG(k) = beatInfo(k).segmentStartIndex + beatInfo(k).rIndex - 1;
end

% Calculate RR_Prev and RR_Post (in seconds)
rr_intervals = diff(allRIndicesInECG) / fs;
X_context(2:end, 1) = rr_intervals; % RR_Prev for beat 2 to N
X_context(1:end-1, 2) = rr_intervals; % RR_Post for beat 1 to N-1

% --- Step 2: Enhanced context feature extraction (if enabled) ---
if params.enhancedContext && numBeats >= 5
    fprintf('    Calculating enhanced context features...\n');
    
    % Calculate heart rate variability metrics (standard deviation of RR intervals in local window)
    for i = 1:numBeats
        windowStart = max(1, i-2);
        windowEnd = min(numBeats-1, i+1);
        localRRs = rr_intervals(windowStart:windowEnd);
        
        if length(localRRs) >= 2
            X_context(i, 3) = std(localRRs); % HR_Variability
        else
            X_context(i, 3) = 0;
        end
    end
    
    % Calculate rhythm stability (coefficient of variation based on local RR interval)
    for i = 1:numBeats
        windowStart = max(1, i-2);
        windowEnd = min(numBeats-1, i+1);
        localRRs = rr_intervals(windowStart:windowEnd);
        
        if length(localRRs) >= 2 && mean(localRRs) > 0
            cv = std(localRRs) / mean(localRRs); % Coefficient of variation
            X_context(i, 4) = 1 / (1 + cv); % Stability index (closer to 1 means more stable)
        else
            X_context(i, 4) = 0.5; % Default medium stability
        end
    end
end

% --- Step 3: Waveform preprocessing and data quality detection ---
fprintf('    Processing waveform data and performing quality check...\n');
qualityFlags = true(numBeats, 1); % Quality flags

for i = 1:numBeats
    currentBeat = beatInfo(i);
    segment = currentBeat.segment;
    
    % a. Standardize waveform length
    if ~isempty(segment)
        segment = segment(:)'; % Ensure it's a row vector
        
        % Data quality detection
        if params.enableQualityCheck
            % Check 1: Waveform length validity
            if length(segment) < standardLength/4
                qualityFlags(i) = false;
                fprintf('      Warning: Heartbeat %d waveform too short (%d samples)\n', i, length(segment));
            end
            
            % Check 2: Signal amplitude range check (to avoid extreme outliers)
            segmentRange = max(segment) - min(segment);
            segmentMean = mean(segment);
            segmentStd = std(segment);

            % Smarter anomaly detection
            if segmentRange < 1e-6 || segmentRange > 50 || ... % Amplitude range check
               abs(segmentMean) > 10 || ...                    % Mean value anomaly check
               segmentStd < 1e-6 || segmentStd > 20            % Standard deviation anomaly check
                qualityFlags(i) = false;
                fprintf('      Warning: Heartbeat %d signal anomaly (range: %.6f, mean: %.6f, std: %.6f)\n', ...
                    i, segmentRange, segmentMean, segmentStd);
            end
            
            % Check 3: NaN or Inf check
            if any(isnan(segment)) || any(isinf(segment))
                qualityFlags(i) = false;
                fprintf('      Warning: Heartbeat %d contains NaN or Inf values\n', i);
            end
        end
        
        % Resampling
        if length(segment) == standardLength
            resampled_segment = segment;
        else
            try
                resampled_segment = resample(segment, standardLength, length(segment));
            catch
                % If resampling fails, use linear interpolation as a fallback
                resampled_segment = interp1(1:length(segment), segment, ...
                    linspace(1, length(segment), standardLength), 'linear', 'extrap');
                if params.enableQualityCheck
                    fprintf('      Note: Heartbeat %d resampled using linear interpolation\n', i);
                end
            end
        end
        
        X_waveforms{i} = resampled_segment;
    else
        % If the original segment is empty, create a zero vector and mark quality issue
        X_waveforms{i} = zeros(1, standardLength);
        if params.enableQualityCheck
            qualityFlags(i) = false;
            fprintf('      Warning: Heartbeat %d original segment is empty\n', i);
        end
    end
    
    % b. Extract labels
    Y_labels_cell{i} = char(currentBeat.beatType);
end

% --- Step 4: Multi-scale waveform normalization ---
fprintf('    Applying %s normalization method...\n', params.normalizationMethod);

for i = 1:numBeats
    if ~qualityFlags(i), continue; end % Skip data with poor quality
    
    waveform = X_waveforms{i};
    
    switch params.normalizationMethod
        case 'zscore'
            % Z-score normalization (default method)
            meanVal = mean(waveform);
            stdVal = std(waveform);
            if stdVal > 1e-10
                X_waveforms{i} = (waveform - meanVal) / stdVal;
            else
                X_waveforms{i} = zeros(1, standardLength);
            end
            
        case 'minmax'
            % Min-Max normalization to [0,1]
            minVal = min(waveform);
            maxVal = max(waveform);
            if abs(maxVal - minVal) > 1e-10
                X_waveforms{i} = (waveform - minVal) / (maxVal - minVal);
            else
                X_waveforms{i} = zeros(1, standardLength);
            end
            
        case 'robust'
            % Robust normalization (using median and interquartile range)
            medianVal = median(waveform);
            iqr = prctile(waveform, 75) - prctile(waveform, 25);
            if iqr > 1e-10
                X_waveforms{i} = (waveform - medianVal) / iqr;
            else
                X_waveforms{i} = zeros(1, standardLength);
            end
            
        otherwise
            warning('Unknown normalization method: %s, using default Z-score method', params.normalizationMethod);
            meanVal = mean(waveform);
            stdVal = std(waveform);
            if stdVal > 1e-10
                X_waveforms{i} = (waveform - meanVal) / stdVal;
            else
                X_waveforms{i} = zeros(1, standardLength);
            end
    end
end

% --- Step 5: Context feature normalization ---
fprintf('    Normalizing context features...\n');
for col = 1:contextDim
    contextCol = X_context(:, col);
    validIdx = ~isnan(contextCol) & qualityFlags;
    
    if sum(validIdx) > 1
        meanVal = mean(contextCol(validIdx));
        stdVal = std(contextCol(validIdx));
        
        if stdVal > 1e-10
            X_context(validIdx, col) = (contextCol(validIdx) - meanVal) / stdVal;
        else
            X_context(validIdx, col) = 0;
        end
    end
end

% Replace remaining NaN values with 0 (corresponding to normalized mean value)
X_context(isnan(X_context)) = 0;

% --- Step 6: Quality filtering and final output ---
if params.enableQualityCheck
    validCount = sum(qualityFlags);
    invalidCount = sum(~qualityFlags);
    
    if invalidCount > 0
        fprintf('    Data quality check complete: %d valid samples, %d invalid samples marked\n', validCount, invalidCount);
        fprintf('    Note: Invalid samples are still retained in the dataset but marked, can be filtered in subsequent steps\n');
        
        % Set flags for invalid samples (can be filtered out in later processing)
        for i = 1:numBeats
            if ~qualityFlags(i)
                X_waveforms{i} = zeros(1, standardLength); % Set to zero vector
                X_context(i, :) = 0; % Set to zero features
            end
        end
    else
        fprintf('    All %d samples passed the quality check\n', numBeats);
    end
end

% Format output
Y_labels = categorical(Y_labels_cell);

fprintf('    Deep learning data preparation complete: Waveform dimension[%d x %d], Context dimension[%d x %d], Label count[%d]\n', ...
    length(X_waveforms), standardLength, size(X_context,1), size(X_context,2), length(Y_labels));

if params.enhancedContext
    fprintf('    Enhanced context features: %s\n', strjoin(contextNames, ', '));
end

end