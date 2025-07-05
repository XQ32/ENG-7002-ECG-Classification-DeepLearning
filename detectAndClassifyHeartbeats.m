function [heartbeatSegments, beatInfo] = detectAndClassifyHeartbeats(ecg, ATRTIMED, ANNOTD, fs)
% detectAndClassifyHeartbeats - Detects all heartbeats in an ECG signal based on annotation data, classifies them, and extracts initial information.
%
% Inputs:
%   ecg - Filtered ECG signal (column vector)
%   ATRTIMED - Time points of annotations (in seconds) (column vector)
%   ANNOTD - Annotation labels (string cell array), e.g., 'Normal': Normal, 'PVC': PVC, 'PAC': PAC, 'Other': Other
%   fs - Sampling frequency (Hz)
%
% Outputs:
%   heartbeatSegments - Cell array, containing the ECG segment for each heartbeat (in column vector form)
%   beatInfo - Struct array, containing heartbeat classification and feature point information
%     .beatType - Heartbeat type ('Normal': Normal, 'PVC': PVC, 'PAC': PAC, 'Other': Other)
%     .segment - Extracted ECG segment
%     .rIndex - Index of the R-wave within the segment
%     .pIndex - Index of the P-wave within the segment
%     .qIndex - Index of the Q-wave within the segment
%     .sIndex - Index of the S-wave within the segment
%     .tIndex - Index of the T-wave within the segment
%     .pEnd - End point of the P-wave within the segment
%     .tEnd - End point of the T-wave within the segment
%     .segmentStartIndex - Starting index of the segment in the original ECG signal

% Ensure ecg is a column vector
ecg = ecg(:);

% Initialize output variables - ensure they are column vectors
heartbeatSegments = cell(3000000, 1);  % Pre-allocate a 3 million row empty cell array

% Create an initial struct template
initial_beat = struct('beatType', 'Other', 'segment', [], 'rIndex', 0, 'pIndex', NaN, ...
                  'qIndex', NaN, 'sIndex', NaN, 'tIndex', NaN, 'pEnd', NaN, 'tEnd', NaN, ...
                  'segmentStartIndex', 0);
                  
% Pre-allocate memory to hold up to 3 million heartbeat infos
beatInfo = repmat(initial_beat, 3000000, 1);
                  
% Initialize heartbeat counters
heartbeat_count = 0;
beat_info_count = 0;

% Get all heartbeat time points and types
beatTimes_seconds = ATRTIMED(:); % Ensure it's a column vector, and already in seconds
originalBeatTypes = ANNOTD(:);   % Ensure it's a column vector, a cell array containing string labels

% Convert time points to sample indices (relative to the start of the ECG signal)
beatIndices = round(beatTimes_seconds * fs); 

% Filter for valid annotation types and indices
validBeatTypeCodes = {'PVC', 'Other'}; % The two classes we are interested in: PVC and Other (PAC has been merged into Other)
isConsideredType = ismember(originalBeatTypes, validBeatTypeCodes);

% beatTimes_seconds_filtered = beatTimes_seconds(isConsideredType);
beatTypes_filtered = originalBeatTypes(isConsideredType);
beatIndices_filtered = beatIndices(isConsideredType);

% Further filter out invalid indices that are outside the signal length range
validIdx = beatIndices_filtered > 0 & beatIndices_filtered <= length(ecg);
finalBeatTypes = beatTypes_filtered(validIdx);
finalBeatIndices = beatIndices_filtered(validIdx);

try
    % Directly segment heartbeats based on original annotation indices
    fprintf('Starting to segment heartbeats based on original annotation indices...\n');
    
    % Heartbeat segment length: 0.4 seconds before and after the R-wave
    segmentLength = round(0.4 * fs);
    
    % Process each original annotated R-wave
for i = 1:length(finalBeatIndices)
    % Get the type and index in the original ECG of the current heartbeat
        currentBeatType = finalBeatTypes(i);
        currentRIndex = finalBeatIndices(i);
        
        % Determine the start and end positions of the segment
        segmentStart = max(1, currentRIndex - segmentLength);
        segmentEnd = min(length(ecg), currentRIndex + segmentLength);
        
        % Check if the segment length meets expectations
        if (segmentEnd - segmentStart + 1) < (2 * segmentLength * 0.9)
            fprintf('Warning: R-wave segment at index %d is too short, skipping.\n', currentRIndex);
            continue;
        end
    
        % Extract the heartbeat segment
        segment = ecg(segmentStart:segmentEnd);
        
        % Add the heartbeat using a counter
        heartbeat_count = heartbeat_count + 1;
        heartbeatSegments{heartbeat_count, 1} = segment;
        
        % Calculate the relative position of the R-wave in the current segment
        rIndexInSegment = currentRIndex - segmentStart + 1;
        
        % Apply waveform detection algorithm to the current heartbeat segment
        [pIndexInSegment, qIndexInSegment, sIndexInSegment, tIndexInSegment, pEndInSegment, tEndInSegment] = ...
            detectWaveformsInSegment(segment, rIndexInSegment, fs, currentBeatType);
        
        % Create the heartbeat information struct
        beat = struct('beatType', currentBeatType, ...
                      'segment', segment, ...
                      'rIndex', rIndexInSegment, ...
                      'pIndex', pIndexInSegment, ...
                      'qIndex', qIndexInSegment, ...
                      'sIndex', sIndexInSegment, ...
                      'tIndex', tIndexInSegment, ...
                      'pEnd', pEndInSegment, ...
                      'tEnd', tEndInSegment, ...
                      'segmentStartIndex', segmentStart);
                      
        beat_info_count = beat_info_count + 1;
        beatInfo(beat_info_count) = beat;
    end
    
    fprintf('Heartbeat segmentation complete.\n');
    
catch ME
    fprintf('Error during heartbeat segmentation process: %s\n', ME.message);
    % No longer throws an error, but returns an empty result
    heartbeatSegments = cell(0, 1);  % Return an empty cell array, not the pre-allocated one
    beatInfo = initial_beat([]); % Create an empty struct array but keep field names
    fprintf('Returning empty results due to error, continuing to the next record.\n');
    return;  % Return early, skipping the subsequent trimming step
end

% Return only the actually used parts
heartbeatSegments = heartbeatSegments(1:heartbeat_count);
beatInfo = beatInfo(1:beat_info_count);

% Final statistics
if beat_info_count > 0
    types_extracted = {beatInfo.beatType};
    fprintf('Processing complete, a total of %d heartbeats were extracted and recorded.\n', beat_info_count);
    fprintf('  Extracted PVC beats (Type PVC): %d\n', sum(strcmp(types_extracted, 'PVC')));
    fprintf('  Extracted Other beats (Type Other): %d\n', sum(strcmp(types_extracted, 'Other')));
else
    fprintf('Processing complete, no valid heartbeat information could be extracted.\n');
end

fprintf('Returning %d heartbeat segments and info (trimmed from pre-allocated array)\n', heartbeat_count);

end

% Sub-function: perform waveform detection on a single heartbeat segment
function [pIndex, qIndex, sIndex, tIndex, pEnd, tEnd] = detectWaveformsInSegment(segment, rIndex, fs, beatType)
% detectWaveformsInSegment - Uses alternative algorithms to detect P, Q, S, T waves in a single heartbeat segment
%
% Inputs:
%   segment - ECG segment of a heartbeat
%   rIndex - Position of the R-wave in the segment
%   fs - Sampling frequency
%   beatType - Heartbeat type
%
% Outputs:
%   pIndex, qIndex, sIndex, tIndex - Position of each waveform in the segment
%   pEnd, tEnd - End positions of the P and T waves

    % Initialize output variables
    pIndex = NaN;
    qIndex = NaN;
    sIndex = NaN;
    tIndex = NaN;
    pEnd = NaN;
    tEnd = NaN;
    
    % Ensure segment is a column vector
    segment = segment(:);
    segmentLength = length(segment);
    
    try
        % Method 1: Basic local extremum detection
        % Q-wave detection - search for local minimum within 100ms before R-wave
        qSearchStart = max(1, rIndex - round(0.1*fs));
        qSearchEnd = rIndex - 1;
        if qSearchEnd > qSearchStart
            qSearchWindow = qSearchStart:qSearchEnd;
            if length(qSearchWindow) >= 3
                [~, minIdx] = min(segment(qSearchWindow));
                qIndex = qSearchWindow(minIdx);
            end
        end
        
        % S-wave detection - search for local minimum within 100ms after R-wave
        sSearchStart = rIndex + 1;
        sSearchEnd = min(segmentLength, rIndex + round(0.1*fs));
        if sSearchEnd > sSearchStart
            sSearchWindow = sSearchStart:sSearchEnd;
            if length(sSearchWindow) >= 3
                [~, minIdx] = min(segment(sSearchWindow));
                sIndex = sSearchWindow(minIdx);
            end
        end
        
        % T-wave detection - use improved algorithm
        [tIndex, tEnd] = detectTWave(segment, rIndex, sIndex, fs);
        
        % P-wave detection - use improved algorithm
        [pIndex, pEnd] = detectPWave(segment, rIndex, qIndex, fs);
        
        % Method 2: Use more complex algorithms for abnormal heartbeats
        if strcmp(beatType, 'PVC') || strcmp(beatType, 'PAC') || strcmp(beatType, 'Other')
            % Use P/T wave detection based on a triangular filter (from detectionFunctions)
            [pIndex_alt, tIndex_alt] = detectPTWavesWithTriangularFilter(segment, rIndex, fs);
            
            % If the basic method did not detect it, use the alternative result
            if isnan(pIndex) && ~isnan(pIndex_alt)
                pIndex = pIndex_alt;
                % Recalculate P-wave end point
                if ~isnan(qIndex)
                    pEnd = round((pIndex + qIndex) / 2);
                else
                    pEnd = min(segmentLength, pIndex + round(0.07 * fs));
                end
                pEnd = max(1, min(pEnd, segmentLength));
            end
            
            if isnan(tIndex) && ~isnan(tIndex_alt)
                tIndex = tIndex_alt;
                tEnd = min(segmentLength, tIndex + round(0.1 * fs));
            end
            
            % Use enhanced Q/S wave detection
            [qIndex_alt, sIndex_alt] = detectQSWavesEnhanced(segment, rIndex, fs);
            
            % If the basic method did not detect it, use the alternative result
            if isnan(qIndex) && ~isnan(qIndex_alt)
                qIndex = qIndex_alt;
            end
            if isnan(sIndex) && ~isnan(sIndex_alt)
                sIndex = sIndex_alt;
            end
        end
        
    catch ME
        fprintf('Warning: An error occurred during waveform detection: %s\n', ME.message);
        % Return NaN values on error
    end
end

% P-wave detection sub-function
function [pIndex, pEnd] = detectPWave(segment, rIndex, qIndex, fs)
    pIndex = NaN;
    pEnd = NaN;
    segmentLength = length(segment);
    
    % Determine the search range
    if ~isnan(qIndex)
        pSearchEnd = qIndex - round(0.04*fs); % End 40ms before Q-wave
    else
        pSearchEnd = rIndex - round(0.08*fs); % End 80ms before R-wave
    end
    pSearchStart = max(1, rIndex - round(0.25*fs)); % Start 250ms before R-wave
    
    if pSearchEnd > pSearchStart
        pSearchWindow = pSearchStart:pSearchEnd;
        if length(pSearchWindow) >= 5
            % Method 1: Directly find the maximum value
            if length(pSearchWindow) > 10
                smoothed = movmean(segment(pSearchWindow), 5);
                [~, maxIdx] = max(smoothed);
            else
                [~, maxIdx] = max(segment(pSearchWindow));
            end
            pIndex = pSearchWindow(maxIdx);
            
            % Method 2: Find a relatively high peak (to avoid noise interference)
            threshold = mean(segment(pSearchWindow)) + 0.3 * std(segment(pSearchWindow));
            candidateIndices = find(segment(pSearchWindow) > threshold);
            if ~isempty(candidateIndices)
                [~, maxIdx] = max(segment(pSearchWindow(candidateIndices)));
                pIndex_candidate = pSearchWindow(candidateIndices(maxIdx));
                
                % Choose a more reasonable P-wave position
                if abs(pIndex_candidate - (rIndex - round(0.16*fs))) < abs(pIndex - (rIndex - round(0.16*fs)))
                    pIndex = pIndex_candidate;
                end
            end
            
            % Estimate P-wave end point
            if ~isnan(qIndex)
                pEnd = round((pIndex + qIndex) / 2);
            else
                pEnd = min(segmentLength, pIndex + round(0.07 * fs)); % P-wave width is about 70ms
            end
            pEnd = max(1, min(pEnd, segmentLength));
        end
    end
end

% T-wave detection sub-function
function [tIndex, tEnd] = detectTWave(segment, rIndex, sIndex, fs)
    tIndex = NaN;
    tEnd = NaN;
    segmentLength = length(segment);
    
    % Determine the search range
    if ~isnan(sIndex)
        tSearchStart = sIndex + round(0.05*fs); % Start 50ms after S-wave
    else
        tSearchStart = rIndex + round(0.15*fs); % Start 150ms after R-wave
    end
    tSearchEnd = min(segmentLength, rIndex + round(0.45*fs));
    
    if tSearchEnd > tSearchStart
        tSearchWindow = tSearchStart:tSearchEnd;
        if length(tSearchWindow) >= 5
            % Method 1: Directly find the maximum value
            if length(tSearchWindow) > 10
                smoothed = movmean(segment(tSearchWindow), 5);
                [~, maxIdx] = max(smoothed);
            else
                [~, maxIdx] = max(segment(tSearchWindow));
            end
            tIndex = tSearchWindow(maxIdx);
            
            % Method 2: Find a peak that matches T-wave characteristics
            % T-wave usually appears 200-350ms after the R-wave
            idealTPosition = rIndex + round(0.275*fs); % Ideal T-wave position
            idealTWindow = max(tSearchStart, idealTPosition - round(0.075*fs)):...
                          min(tSearchEnd, idealTPosition + round(0.075*fs));
            
            if ~isempty(idealTWindow)
                [~, maxIdxIdeal] = max(segment(idealTWindow));
                tIndex_ideal = idealTWindow(maxIdxIdeal);
                
                % If the T-wave at the ideal position is significant enough, prefer it
                if segment(tIndex_ideal) > 0.7 * segment(tIndex)
                    tIndex = tIndex_ideal;
                end
            end
            
            % Estimate T-wave end point: about 100ms after T-wave peak
            tEnd = min(segmentLength, tIndex + round(0.1 * fs));
        end
    end
end

% P/T wave detection based on triangular filter (idea from detectionFunctions)
function [pIndex, tIndex] = detectPTWavesWithTriangularFilter(segment, rIndex, fs)
    pIndex = NaN;
    tIndex = NaN;
    
    try
        % Create a triangular filter (0.1s width)
        triangular_duration = round(0.1 * fs);
        init_filter_coeffs = -triangular_duration/2:triangular_duration/2;
        triangular_filter = ((-abs(init_filter_coeffs) + triangular_duration/2 * ones(1, triangular_duration+1)) / (triangular_duration/2))';
        
        % Remove the influence of the QRS complex
        qrs_radius = round(triangular_duration/2);
        segment_for_pt = segment;
        
        % Zero out around the R-wave
        zeroStart = max(1, rIndex - qrs_radius);
        zeroEnd = min(length(segment), rIndex + qrs_radius);
        segment_for_pt(zeroStart:zeroEnd) = 0;
        
        % Apply the triangular filter
        if length(segment_for_pt) >= length(triangular_filter)
            correlated_data = conv(segment_for_pt, triangular_filter, 'same');
            
            % Find P-wave (before R-wave)
            pSearchStart = max(1, rIndex - round(0.3*fs));
            pSearchEnd = rIndex - round(0.05*fs);
            if pSearchEnd > pSearchStart
                pSearchWindow = pSearchStart:pSearchEnd;
                [~, maxIdx] = max(correlated_data(pSearchWindow));
                pIndex = pSearchWindow(maxIdx);
            end
            
            % Find T-wave (after R-wave)
            tSearchStart = rIndex + round(0.1*fs);
            tSearchEnd = min(length(segment), rIndex + round(0.5*fs));
            if tSearchEnd > tSearchStart
                tSearchWindow = tSearchStart:tSearchEnd;
                [~, maxIdx] = max(correlated_data(tSearchWindow));
                tIndex = tSearchWindow(maxIdx);
            end
        end
    catch
        % If an error occurs, return NaN
    end
end

% Enhanced Q/S wave detection
function [qIndex, sIndex] = detectQSWavesEnhanced(segment, rIndex, fs)
    qIndex = NaN;
    sIndex = NaN;
    
    try
        % Q-wave detection - use derivative and morphological methods
        qSearchStart = max(1, rIndex - round(0.12*fs));
        qSearchEnd = rIndex - 1;
        
        if qSearchEnd > qSearchStart
            qSearchWindow = qSearchStart:qSearchEnd;
            
            % Method 1: Find local minimum
            [~, minIdx] = min(segment(qSearchWindow));
            qIndex_method1 = qSearchWindow(minIdx);
            
            % Method 2: Find point after maximum negative derivative
            if length(qSearchWindow) > 3
                diff_signal = diff(segment(qSearchWindow));
                [~, minDiffIdx] = min(diff_signal);
                qIndex_method2 = qSearchWindow(minDiffIdx + 1);
                
                % Choose a more reasonable Q-wave position (usually 40-80ms before R-wave)
                ideal_q_pos = rIndex - round(0.06*fs);
                if abs(qIndex_method2 - ideal_q_pos) < abs(qIndex_method1 - ideal_q_pos)
                    qIndex = qIndex_method2;
                else
                    qIndex = qIndex_method1;
                end
            else
                qIndex = qIndex_method1;
            end
        end
        
        % S-wave detection - use derivative and morphological methods
        sSearchStart = rIndex + 1;
        sSearchEnd = min(length(segment), rIndex + round(0.12*fs));
        
        if sSearchEnd > sSearchStart
            sSearchWindow = sSearchStart:sSearchEnd;
            
            % Method 1: Find local minimum
            [~, minIdx] = min(segment(sSearchWindow));
            sIndex_method1 = sSearchWindow(minIdx);
            
            % Method 2: Find point after maximum negative derivative
            if length(sSearchWindow) > 3
                diff_signal = diff(segment(sSearchWindow));
                [~, minDiffIdx] = min(diff_signal);
                sIndex_method2 = sSearchWindow(minDiffIdx + 1);
                
                % Choose a more reasonable S-wave position (usually 20-60ms after R-wave)
                ideal_s_pos = rIndex + round(0.04*fs);
                if abs(sIndex_method2 - ideal_s_pos) < abs(sIndex_method1 - ideal_s_pos)
                    sIndex = sIndex_method2;
                else
                    sIndex = sIndex_method1;
                end
            else
                sIndex = sIndex_method1;
            end
        end
        
    catch
        % If an error occurs, return NaN
    end
end

% Simplified peak detection sub-function
function peaks = detectPeaksInSegment(data)
% detectPeaksInSegment - Simplified peak detection algorithm
    peaks = [];
    data = data(:);
    n = length(data);
    
    if n < 3
        return;
    end
    
    % Find local maxima
    for i = 2:n-1
        if data(i) > data(i-1) && data(i) > data(i+1) && data(i) > 0.1*max(data)
            peaks = [peaks; i];
        end
    end
end