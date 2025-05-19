function [heartbeatSegments, beatInfo] = detectAndClassifyHeartbeats(ecg, ATRTIMED, ANNOTD, fs)
% detectAndClassifyHeartbeats - Detects all heartbeats in the ECG signal based on annotation data, classifies them, and extracts preliminary information
%
% Input:
%   ecg - Filtered ECG signal (column vector)
%   ATRTIMED - Annotation time points (seconds) (column vector)
%   ANNOTD - Annotation labels (numerical codes, column vector), e.g., 1:Normal, 5:PVC, 8:PAC
%   fs - Sampling frequency (Hz)
%
% Output:
%   heartbeatSegments - Cell array, containing ECG segments for each heartbeat (as column vectors)
%   beatInfo - Struct array, containing heartbeat classification and feature point information
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

% Ensure ecg is a column vector
ecg = ecg(:);

% Initialize output variables - ensure they are column vectors
heartbeatSegments = cell(3000000, 1);  % Preallocate a cell array for 3 million rows

% Create initial struct template
initial_beat = struct('beatType', 0, 'segment', [], 'rIndex', 0, 'pIndex', NaN, ...
                  'qIndex', NaN, 'sIndex', NaN, 'tIndex', NaN, 'pEnd', NaN, 'tEnd', NaN, ...
                  'segmentStartIndex', 0);
                  
% Preallocate memory for 3 million heartbeat info structs
beatInfo = repmat(initial_beat, 3000000, 1);
                  
% Initialize heartbeat counters
heartbeat_count = 0;
beat_info_count = 0;

% Get all heartbeat times and types
beatTimes_seconds = ATRTIMED(:); % Ensure it's a column vector, already in seconds
originalBeatTypes = ANNOTD(:);   % Ensure it's a column vector, already numerical codes

% Convert time points to sample indices (relative to the start of the ECG signal)
beatIndices = round(beatTimes_seconds * fs); 

% Filter valid annotation types and indices
validBeatTypeCodes = [1, 5, 8]; % Type codes we are interested in
isConsideredType = ismember(originalBeatTypes, validBeatTypeCodes);

beatTimes_seconds_filtered = beatTimes_seconds(isConsideredType);
beatTypes_filtered = originalBeatTypes(isConsideredType);
beatIndices_filtered = beatIndices(isConsideredType);

% Further filter out invalid indices outside the signal length range
validIdx = beatIndices_filtered > 0 & beatIndices_filtered <= length(ecg);
% beatTimes_seconds_final = beatTimes_seconds_filtered(validIdx);
finalBeatTypes = beatTypes_filtered(validIdx);
finalBeatIndices = beatIndices_filtered(validIdx);

try
    % Part 1: Apply heplab_T_detect_MTEO algorithm to the entire ECG signal to detect all waveform points
    fprintf('Starting waveform point detection for the entire ECG signal...\n');
    [R_points, Q_points, S_points, T_points, P_points] = heplab_T_detect_MTEO(ecg, fs, false);
    
    % Output detection result statistics
    fprintf('Waveform point detection completed, detected:\n');
    fprintf('  R-wave points: %d\n', size(R_points, 1));
    fprintf('  Q-wave points: %d\n', size(Q_points, 1));
    fprintf('  S-wave points: %d\n', size(S_points, 1));
    fprintf('  T-wave points: %d\n', size(T_points, 1));
    fprintf('  P-wave points: %d\n', size(P_points, 1));
    
    % Part 2: Segment heartbeats based on original annotation indices
    fprintf('Starting heartbeat segmentation based on original annotation indices...\n');
    
    % Heartbeat segment length: 0.5 seconds before and after R-wave
    segmentLength = round(0.4 * fs);
    
    % Process each original annotated R-wave
    for i = 1:length(finalBeatIndices)
        % Get the type and index in the original ECG of the current heartbeat
        currentBeatType = finalBeatTypes(i);
        currentRIndex = finalBeatIndices(i);
        
        % Determine the start and end positions of the segment
        segmentStart = max(1, currentRIndex - segmentLength);
        segmentEnd = min(length(ecg), currentRIndex + segmentLength);
        
        % Check if the segment length is as expected
        if (segmentEnd - segmentStart + 1) < (2 * segmentLength * 0.9)
            fprintf('Warning: R-wave at index %d has insufficient segment length, skipping.\n', currentRIndex);
            continue;
        end
    
    % Extract heartbeat segment
    segment = ecg(segmentStart:segmentEnd);
    
    % Add heartbeat using counter
    heartbeat_count = heartbeat_count + 1;
    heartbeatSegments{heartbeat_count, 1} = segment;
    
    % Calculate the relative position of the R-wave in the current segment
    rIndexInSegment = currentRIndex - segmentStart + 1;
    
    % Initialize indices of other waveform feature points in the segment
    pIndexInSegment = NaN;
    qIndexInSegment = NaN;
    sIndexInSegment = NaN;
    tIndexInSegment = NaN;
    pEndInSegment = NaN;
    tEndInSegment = NaN;
    
        % Find the Q-wave corresponding to the current segment
        validQ = Q_points(Q_points(:,1) < currentRIndex & Q_points(:,1) > currentRIndex - round(0.2*fs), 1);
        if ~isempty(validQ)
            [~, idx] = max(validQ);  % Select the Q-point closest to the R-wave
            qInECG = validQ(idx);
            if qInECG >= segmentStart && qInECG <= segmentEnd
                qIndexInSegment = qInECG - segmentStart + 1;
            end
        end
        
        % Alternative Q-wave detection algorithm - if the main method did not find a Q-wave
        if isnan(qIndexInSegment) && (currentBeatType == 5 || currentBeatType == 8) % Only apply alternative algorithm to non-normal heartbeats
            % Search for local minimum as Q-wave within a 100ms window before R-wave
            search_window = max(1, rIndexInSegment - round(0.1*fs)):rIndexInSegment-1;
            if length(search_window) >= 3  % Ensure enough points
                [min_val, min_idx] = min(segment(search_window));
                q_candidate = search_window(min_idx);
                if ~isempty(q_candidate) && q_candidate > 0 && q_candidate < length(segment)
                    qIndexInSegment = q_candidate;
                    fprintf('    Q-wave detected using alternative algorithm, located %dms before R-wave\n', round((rIndexInSegment-qIndexInSegment)*1000/fs));
                end
            end
        end
        
        % Find the S-wave corresponding to the current segment
        validS = S_points(S_points(:,1) > currentRIndex & S_points(:,1) < currentRIndex + round(0.2*fs), 1);
        if ~isempty(validS)
            [~, idx] = min(validS);  % Select the S-point closest to the R-wave
            sInECG = validS(idx);
            if sInECG >= segmentStart && sInECG <= segmentEnd
                sIndexInSegment = sInECG - segmentStart + 1;
            end
        end
        
        % Alternative S-wave detection algorithm - if the main method did not find an S-wave
        if isnan(sIndexInSegment) && (currentBeatType == 5 || currentBeatType == 8)  % Only apply alternative algorithm to non-normal heartbeats
            % Search for local minimum as S-wave within a 100ms window after R-wave
            search_window = (rIndexInSegment+1):min(length(segment), rIndexInSegment + round(0.1*fs));
            if length(search_window) >= 3  % Ensure enough points
                [min_val, min_idx] = min(segment(search_window));
                s_candidate = search_window(min_idx);
                if ~isempty(s_candidate) && s_candidate > 0 && s_candidate < length(segment)
                    sIndexInSegment = s_candidate;
                    fprintf('    S-wave detected using alternative algorithm, located %dms after R-wave\n', round((sIndexInSegment-rIndexInSegment)*1000/fs));
                end
            end
        end
        
        % Find the T-wave corresponding to the current segment - modified to find directly based on R-wave index
        validT = T_points(T_points(:,1) > currentRIndex & T_points(:,1) < currentRIndex + round(0.5*fs), 1);
        if ~isempty(validT)
            [~, idx] = min(validT);  % Select the T-point closest after the R-wave
            tInECG = validT(idx);
            if tInECG >= segmentStart && tInECG <= segmentEnd
                tIndexInSegment = tInECG - segmentStart + 1;
                % Estimate T-wave end point: about 100ms after T-wave peak
                tEndInSegment = min(length(segment), tIndexInSegment + round(0.1 * fs));
            end
        end
        
        % Alternative T-wave detection algorithm - if the main method did not find a T-wave
        if isnan(tIndexInSegment) && (currentBeatType == 5 || currentBeatType == 8)  % Only apply alternative algorithm to non-normal heartbeats
            % Search for local maximum in the 150-450ms range after S-wave or R-wave (if no S-wave)
            if ~isnan(sIndexInSegment)
                start_point = sIndexInSegment;
            else
                start_point = rIndexInSegment;
            end
            
            % Define T-wave search window
            t_start = start_point + round(0.15*fs);
            t_end = min(length(segment), start_point + round(0.45*fs));
            search_window = t_start:t_end;
            
            if length(search_window) >= 5  % Ensure enough points
                % Smooth signal to reduce noise impact
                if length(search_window) > 10
                    smoothed = movmean(segment(search_window), 5);
                    [max_val, max_idx] = max(smoothed);
                    t_candidate = search_window(max_idx);
                else
                    [max_val, max_idx] = max(segment(search_window));
                    t_candidate = search_window(max_idx);
                end
                
                if ~isempty(t_candidate) && t_candidate > 0 && t_candidate < length(segment)
                    tIndexInSegment = t_candidate;
                    % Estimate T-wave end point
                    tEndInSegment = min(length(segment), tIndexInSegment + round(0.1 * fs));
                    fprintf('    T-wave detected using alternative algorithm, located %dms after R-wave\n', round((tIndexInSegment-rIndexInSegment)*1000/fs));
                end
            end
        end
        
        % Find the P-wave corresponding to the current segment - modified to find directly based on R-wave index
        validP = P_points(P_points(:,1) < currentRIndex & P_points(:,1) > currentRIndex - round(0.3*fs), 1);
        if ~isempty(validP)
            [~, idx] = max(validP);  % Select the P-point closest before the R-wave
            pInECG = validP(idx);
            if pInECG >= segmentStart && pInECG <= segmentEnd
                pIndexInSegment = pInECG - segmentStart + 1;
                % Estimate P-wave end point: if Q-wave exists, take midpoint between P-peak and Q-wave
                if ~isnan(qIndexInSegment)
                    pEndInSegment = round((pIndexInSegment + qIndexInSegment) / 2);
                    pEndInSegment = max(1, min(pEndInSegment, length(segment)));
                end
            end
        end
        
        % Alternative P-wave detection algorithm - if the main method did not find a P-wave
        if isnan(pIndexInSegment) && (currentBeatType == 5 || currentBeatType == 8)  % Only apply alternative algorithm to non-normal heartbeats
            % Search for local maximum in the 120-250ms range before R-wave
            if ~isnan(qIndexInSegment)
                end_point = qIndexInSegment;
            else
                end_point = rIndexInSegment;
            end
            
            % Define P-wave search window
            p_end = end_point - round(0.04*fs);  % P-wave should be at least 40ms before Q-wave
            p_start = max(1, end_point - round(0.25*fs));  
            search_window = p_start:p_end;
            
            if length(search_window) >= 5  % Ensure enough points
                % Smooth signal to reduce noise impact
                if length(search_window) > 10
                    smoothed = movmean(segment(search_window), 5);
                    [max_val, max_idx] = max(smoothed);
                    p_candidate = search_window(max_idx);
                else
                    [max_val, max_idx] = max(segment(search_window));
                    p_candidate = search_window(max_idx);
                end
                
                if ~isempty(p_candidate) && p_candidate > 0 && p_candidate < length(segment)
                    pIndexInSegment = p_candidate;
                    
                    % Estimate P-wave end point
                    if ~isnan(qIndexInSegment)
                        pEndInSegment = round((pIndexInSegment + qIndexInSegment) / 2);
                    else
                        pEndInSegment = min(length(segment), pIndexInSegment + round(0.07 * fs)); % P-wave width ~70ms
                    end
                    pEndInSegment = max(1, min(pEndInSegment, length(segment)));
                    fprintf('    P-wave detected using alternative algorithm, located %dms before R-wave\n', round((rIndexInSegment-pIndexInSegment)*1000/fs));
                end
            end
        end
    
    % Create heartbeat information struct
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
    
    fprintf('Heartbeat segmentation completed.\n');
    
catch ME
    fprintf('Error during waveform point detection or heartbeat segmentation: %s\n', ME.message);
    % No longer throw an error, but return empty results
    heartbeatSegments = cell(0, 1);  % Return an empty cell array, not the preallocated one
    beatInfo = initial_beat([]); % Create an empty struct array but keep field names
    fprintf('Returning empty results due to error, continuing to process next record.\n');
    % No longer use rethrow(ME)
    return;  % Return early, skip subsequent trimming steps
end

% Return only the actually used parts
heartbeatSegments = heartbeatSegments(1:heartbeat_count);
beatInfo = beatInfo(1:beat_info_count);

% Final statistics
if beat_info_count > 0
    types_extracted = [beatInfo.beatType];
    fprintf('Processing completed, extracted and recorded %d heartbeats.\n', beat_info_count);
    fprintf('  Extracted Normal heartbeats (Type 1): %d\n', sum(types_extracted == 1));
    fprintf('  Extracted PVC heartbeats (Type 5): %d\n', sum(types_extracted == 5));
    fprintf('  Extracted PAC heartbeats (Type 8): %d\n', sum(types_extracted == 8));
else
    fprintf('Processing completed, failed to extract any valid heartbeat information.\n');
end

fprintf('Returning %d heartbeat segments and info (trimmed from preallocated 3 million rows)\n', heartbeat_count);

end