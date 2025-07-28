function [heartbeatSegments, beatInfo] = detectAndClassifyHeartbeats(ecg, ATRTIMED, ANNOTD, fs)
% detectAndClassifyHeartbeats - Detects all heartbeats in an ECG signal based on annotation data, classifies them, and extracts initial information.
%
% Input:
%   ecg - Filtered ECG signal (column vector)
%   ATRTIMED - Annotation time points (seconds) (column vector)
%   ANNOTD - Annotation labels (string cell array), e.g., 'Normal': Normal, 'PVC': PVC, 'PAC': PAC, 'Other': Other
%   fs - Sampling frequency (Hz)
%
% Output:
%   heartbeatSegments - Cell array containing the ECG segment for each heartbeat (in column vector form)
%   beatInfo - Structure array containing heartbeat classification and feature point information
%     .beatType - Heartbeat type ('Normal': Normal, 'PVC': PVC, 'PAC': PAC, 'Other': Other)
%     .segment - Extracted ECG segment
%     .rIndex - Index of the R-wave within the segment
%     .pIndex - Index of the P-wave within the segment
%     .qIndex - Index of the Q-wave within the segment
%     .sIndex - Index of the S-wave within the segment
%     .tIndex - Index of the T-wave within the segment
%     .pEnd - Index of the P-wave end point within the segment
%     .tEnd - Index of the T-wave end point within the segment
%     .segmentStartIndex - Start index of the segment in the original ECG signal

% Ensure ecg is a column vector
ecg = ecg(:);

% Initialize output variables - ensure they are column vectors
heartbeatSegments = cell(3000000, 1);  % Pre-allocate a cell array for 3 million rows

% Create initial structure template
initial_beat = struct('beatType', 'Other', 'segment', [], 'rIndex', 0, 'pIndex', NaN, ...
                  'qIndex', NaN, 'sIndex', NaN, 'tIndex', NaN, 'pEnd', NaN, 'tEnd', NaN, ...
                  'segmentStartIndex', 0);
                  
% Pre-allocate memory to hold information for 3 million heartbeats
beatInfo = repmat(initial_beat, 3000000, 1);
                  
% Initialize heartbeat counters
heartbeat_count = 0;
beat_info_count = 0;

% Get all heartbeat time points and types
beatTimes_seconds = ATRTIMED(:); % Ensure it's a column vector, and it's already in seconds
originalBeatTypes = ANNOTD(:);   % Ensure it's a column vector, cell array containing string labels

% Convert time points to sample indices (relative to the start of the ECG signal)
beatIndices = round(beatTimes_seconds * fs); 

% Filter valid annotation types and indices
validBeatTypeCodes = {'PVC', 'Other'}; % The two classes we care about: PVC and Other (PAC is merged into Other)
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
    % Get the type and index in the original ECG for the current heartbeat
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
    
        % Extract the heartbeat segment
        segment = ecg(segmentStart:segmentEnd);
        
        % Add heartbeat using a counter
        heartbeat_count = heartbeat_count + 1;
        heartbeatSegments{heartbeat_count, 1} = segment;
        
        % Calculate the relative position of the R-wave in the current segment
        rIndexInSegment = currentRIndex - segmentStart + 1;
        
        % Apply waveform detection algorithm to the current heartbeat segment
        [pIndexInSegment, qIndexInSegment, sIndexInSegment, tIndexInSegment, pEndInSegment, tEndInSegment] = ...
            detectWaveformsInSegment(segment, rIndexInSegment, fs, currentBeatType);
        
        % Create heartbeat information structure
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
    fprintf('Error during heartbeat segmentation: %s\n', ME.message);
    % Do not throw an error, but return empty results
    heartbeatSegments = cell(0, 1);  % Return an empty cell array, not the pre-allocated one
    beatInfo = initial_beat([]); % Create an empty structure array but keep the field names
    fprintf('Due to the error, empty results will be returned, and the next record will be processed.\n');
    return;  % Return early, skip the subsequent cropping steps
end

% Only return the actually used parts
heartbeatSegments = heartbeatSegments(1:heartbeat_count);
beatInfo = beatInfo(1:beat_info_count);

% Final statistics
if beat_info_count > 0
    types_extracted = {beatInfo.beatType};
    fprintf('Processing complete, a total of %d heartbeats have been extracted and recorded.\n', beat_info_count);
    fprintf('  Extracted PVC heartbeats (Type PVC): %d\n', sum(strcmp(types_extracted, 'PVC')));
    fprintf('  Extracted Other heartbeats (Type Other): %d\n', sum(strcmp(types_extracted, 'Other')));
else
    fprintf('Processing complete, no valid heartbeat information could be extracted.\n');
end

fprintf('Returned %d heartbeat segments and information (trimmed from the pre-allocated array)\n', heartbeat_count);

end

% Subfunction: Detect waveforms in a single heartbeat segment
function [pIndex, qIndex, sIndex, tIndex, pEnd, tEnd] = detectWaveformsInSegment(segment, rIndex, fs, beatType)
% detectWaveformsInSegment - 使用备选算法对单个心拍片段检测P、Q、S、T波
%
% 输入:
%   segment - 心拍ECG片段
%   rIndex - R波在片段中的位置
%   fs - 采样频率
%   beatType - 心拍类型
%
% 输出:
%   pIndex, qIndex, sIndex, tIndex - 各波形在片段中的位置
%   pEnd, tEnd - P波和T波的结束位置

    % 初始化输出变量
    pIndex = NaN;
    qIndex = NaN;
    sIndex = NaN;
    tIndex = NaN;
    pEnd = NaN;
    tEnd = NaN;
    
    % 确保segment是列向量
    segment = segment(:);
    segmentLength = length(segment);
    
    try
        % 方法一：基础的局部极值检测
        % Q波检测 - 在R波前100ms内寻找局部最小值
        qSearchStart = max(1, rIndex - round(0.1*fs));
        qSearchEnd = rIndex - 1;
        if qSearchEnd > qSearchStart
            qSearchWindow = qSearchStart:qSearchEnd;
            if length(qSearchWindow) >= 3
                [~, minIdx] = min(segment(qSearchWindow));
                qIndex = qSearchWindow(minIdx);
            end
        end
        
        % S波检测 - 在R波后100ms内寻找局部最小值
        sSearchStart = rIndex + 1;
        sSearchEnd = min(segmentLength, rIndex + round(0.1*fs));
        if sSearchEnd > sSearchStart
            sSearchWindow = sSearchStart:sSearchEnd;
            if length(sSearchWindow) >= 3
                [~, minIdx] = min(segment(sSearchWindow));
                sIndex = sSearchWindow(minIdx);
            end
        end
        
        % T波检测 - 使用改进的算法
        [tIndex, tEnd] = detectTWave(segment, rIndex, sIndex, fs);
        
        % P波检测 - 使用改进的算法
        [pIndex, pEnd] = detectPWave(segment, rIndex, qIndex, fs);
        
        % 方法二：对于异常心拍使用更复杂的算法
        if strcmp(beatType, 'PVC') || strcmp(beatType, 'PAC') || strcmp(beatType, 'Other')
            % 使用基于三角滤波器的P/T波检测（来自detectionFunctions）
            [pIndex_alt, tIndex_alt] = detectPTWavesWithTriangularFilter(segment, rIndex, fs);
            
            % 如果基础方法没有检测到，使用备选结果
            if isnan(pIndex) && ~isnan(pIndex_alt)
                pIndex = pIndex_alt;
                % 重新计算P波结束点
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
            
            % 使用增强的Q/S波检测
            [qIndex_alt, sIndex_alt] = detectQSWavesEnhanced(segment, rIndex, fs);
            
            % 如果基础方法没有检测到，使用备选结果
            if isnan(qIndex) && ~isnan(qIndex_alt)
                qIndex = qIndex_alt;
            end
            if isnan(sIndex) && ~isnan(sIndex_alt)
                sIndex = sIndex_alt;
            end
        end
        
    catch ME
        fprintf('警告：波形检测过程中出现错误: %s\n', ME.message);
        % 出错时返回NaN值
    end
end

% P波检测子函数
function [pIndex, pEnd] = detectPWave(segment, rIndex, qIndex, fs)
    pIndex = NaN;
    pEnd = NaN;
    segmentLength = length(segment);
    
    % 确定搜索范围
    if ~isnan(qIndex)
        pSearchEnd = qIndex - round(0.04*fs); % Q波前40ms结束
    else
        pSearchEnd = rIndex - round(0.08*fs); % R波前80ms结束
    end
    pSearchStart = max(1, rIndex - round(0.25*fs)); % R波前250ms开始
    
    if pSearchEnd > pSearchStart
        pSearchWindow = pSearchStart:pSearchEnd;
        if length(pSearchWindow) >= 5
            % 方法1：直接寻找最大值
            if length(pSearchWindow) > 10
                smoothed = movmean(segment(pSearchWindow), 5);
                [~, maxIdx] = max(smoothed);
            else
                [~, maxIdx] = max(segment(pSearchWindow));
            end
            pIndex = pSearchWindow(maxIdx);
            
            % 方法2：寻找相对较高的峰值（避免噪声干扰）
            threshold = mean(segment(pSearchWindow)) + 0.3 * std(segment(pSearchWindow));
            candidateIndices = find(segment(pSearchWindow) > threshold);
            if ~isempty(candidateIndices)
                [~, maxIdx] = max(segment(pSearchWindow(candidateIndices)));
                pIndex_candidate = pSearchWindow(candidateIndices(maxIdx));
                
                % 选择更合理的P波位置
                if abs(pIndex_candidate - (rIndex - round(0.16*fs))) < abs(pIndex - (rIndex - round(0.16*fs)))
                    pIndex = pIndex_candidate;
                end
            end
            
            % 估算P波结束点
            if ~isnan(qIndex)
                pEnd = round((pIndex + qIndex) / 2);
            else
                pEnd = min(segmentLength, pIndex + round(0.07 * fs)); % P波宽度约70ms
            end
            pEnd = max(1, min(pEnd, segmentLength));
        end
    end
end

% T波检测子函数
function [tIndex, tEnd] = detectTWave(segment, rIndex, sIndex, fs)
    tIndex = NaN;
    tEnd = NaN;
    segmentLength = length(segment);
    
    % 确定搜索范围
    if ~isnan(sIndex)
        tSearchStart = sIndex + round(0.05*fs); % S波后50ms开始
    else
        tSearchStart = rIndex + round(0.15*fs); % R波后150ms开始
    end
    tSearchEnd = min(segmentLength, rIndex + round(0.45*fs));
    
    if tSearchEnd > tSearchStart
        tSearchWindow = tSearchStart:tSearchEnd;
        if length(tSearchWindow) >= 5
            % 方法1：直接寻找最大值
            if length(tSearchWindow) > 10
                smoothed = movmean(segment(tSearchWindow), 5);
                [~, maxIdx] = max(smoothed);
            else
                [~, maxIdx] = max(segment(tSearchWindow));
            end
            tIndex = tSearchWindow(maxIdx);
            
            % 方法2：寻找符合T波特征的峰值
            % T波通常出现在R波后200-350ms之间
            idealTPosition = rIndex + round(0.275*fs); % 理想T波位置
            idealTWindow = max(tSearchStart, idealTPosition - round(0.075*fs)):...
                          min(tSearchEnd, idealTPosition + round(0.075*fs));
            
            if ~isempty(idealTWindow)
                [~, maxIdxIdeal] = max(segment(idealTWindow));
                tIndex_ideal = idealTWindow(maxIdxIdeal);
                
                % 如果理想位置的T波足够明显，优先选择
                if segment(tIndex_ideal) > 0.7 * segment(tIndex)
                    tIndex = tIndex_ideal;
                end
            end
            
            % 估算T波结束点：T波峰值后约100ms
            tEnd = min(segmentLength, tIndex + round(0.1 * fs));
        end
    end
end

% 基于三角滤波器的P/T波检测（来自detectionFunctions的思想）
function [pIndex, tIndex] = detectPTWavesWithTriangularFilter(segment, rIndex, fs)
    pIndex = NaN;
    tIndex = NaN;
    
    try
        % 创建三角滤波器（0.1秒宽度）
        triangular_duration = round(0.1 * fs);
        init_filter_coeffs = -triangular_duration/2:triangular_duration/2;
        triangular_filter = ((-abs(init_filter_coeffs) + triangular_duration/2 * ones(1, triangular_duration+1)) / (triangular_duration/2))';
        
        % 移除QRS复合波的影响
        qrs_radius = round(triangular_duration/2);
        segment_for_pt = segment;
        
        % 在R波周围清零
        zeroStart = max(1, rIndex - qrs_radius);
        zeroEnd = min(length(segment), rIndex + qrs_radius);
        segment_for_pt(zeroStart:zeroEnd) = 0;
        
        % 应用三角滤波器
        if length(segment_for_pt) >= length(triangular_filter)
            correlated_data = conv(segment_for_pt, triangular_filter, 'same');
            
            % 寻找P波（R波前）
            pSearchStart = max(1, rIndex - round(0.3*fs));
            pSearchEnd = rIndex - round(0.05*fs);
            if pSearchEnd > pSearchStart
                pSearchWindow = pSearchStart:pSearchEnd;
                [~, maxIdx] = max(correlated_data(pSearchWindow));
                pIndex = pSearchWindow(maxIdx);
            end
            
            % 寻找T波（R波后）
            tSearchStart = rIndex + round(0.1*fs);
            tSearchEnd = min(length(segment), rIndex + round(0.5*fs));
            if tSearchEnd > tSearchStart
                tSearchWindow = tSearchStart:tSearchEnd;
                [~, maxIdx] = max(correlated_data(tSearchWindow));
                tIndex = tSearchWindow(maxIdx);
            end
        end
    catch
        % 如果出错，返回NaN
    end
end

% 增强的Q/S波检测
function [qIndex, sIndex] = detectQSWavesEnhanced(segment, rIndex, fs)
    qIndex = NaN;
    sIndex = NaN;
    
    try
        % Q波检测 - 使用导数和形态学方法
        qSearchStart = max(1, rIndex - round(0.12*fs));
        qSearchEnd = rIndex - 1;
        
        if qSearchEnd > qSearchStart
            qSearchWindow = qSearchStart:qSearchEnd;
            
            % 方法1：寻找局部最小值
            [~, minIdx] = min(segment(qSearchWindow));
            qIndex_method1 = qSearchWindow(minIdx);
            
            % 方法2：寻找负向最大导数后的点
            if length(qSearchWindow) > 3
                diff_signal = diff(segment(qSearchWindow));
                [~, minDiffIdx] = min(diff_signal);
                qIndex_method2 = qSearchWindow(minDiffIdx + 1);
                
                % 选择更合理的Q波位置（通常在R波前40-80ms）
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
        
        % S波检测 - 使用导数和形态学方法
        sSearchStart = rIndex + 1;
        sSearchEnd = min(length(segment), rIndex + round(0.12*fs));
        
        if sSearchEnd > sSearchStart
            sSearchWindow = sSearchStart:sSearchEnd;
            
            % 方法1：寻找局部最小值
            [~, minIdx] = min(segment(sSearchWindow));
            sIndex_method1 = sSearchWindow(minIdx);
            
            % 方法2：寻找负向最大导数后的点
            if length(sSearchWindow) > 3
                diff_signal = diff(segment(sSearchWindow));
                [~, minDiffIdx] = min(diff_signal);
                sIndex_method2 = sSearchWindow(minDiffIdx + 1);
                
                % 选择更合理的S波位置（通常在R波后20-60ms）
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
        % 如果出错，返回NaN
    end
end

% 简化的峰值检测子函数
function peaks = detectPeaksInSegment(data)
% detectPeaksInSegment - 简化的峰值检测算法
    peaks = [];
    data = data(:);
    n = length(data);
    
    if n < 3
        return;
    end
    
    % 寻找局部最大值
    for i = 2:n-1
        if data(i) > data(i-1) && data(i) > data(i+1) && data(i) > 0.1*max(data)
            peaks = [peaks; i];
        end
    end
end