function plotHeartbeat(beatInfo, beatIndex, fs)
% plotHeartbeat - Plots a single heartbeat waveform and its characteristic points
%
% Input:
%   beatInfo - Structure array containing heartbeat information
%   beatIndex - Index of the heartbeat to display
%   fs - Sampling frequency (Hz)
%
% Function:
%   Plots and marks the ECG waveform and its P, Q, R, S, T wave characteristic points

% Get the selected heartbeat
selectedBeat = beatInfo(beatIndex);
segment = selectedBeat.segment;

% Create time axis (centered on R-wave)
timeAxis = (0:length(segment)-1) / fs;
% Adjust time axis so that R-wave is at time 0
if ~isnan(selectedBeat.rIndex)
    timeAxis = timeAxis - selectedBeat.rIndex/fs;
end

% Plot results
figure('Name', ['Heartbeat Analysis - Beat Index ' num2str(beatIndex)], 'Position', [100, 100, 1000, 500]);

% Plot ECG signal
plot(timeAxis, segment, 'b', 'LineWidth', 1.5);
hold on;

% Mark R-wave position
if ~isnan(selectedBeat.rIndex)
    scatter(0, segment(selectedBeat.rIndex), 100, 'ro', 'LineWidth', 2, 'DisplayName', 'R-wave');
end

% Mark P-wave position
if ~isnan(selectedBeat.pIndex)
    pTime = timeAxis(selectedBeat.pIndex);
    scatter(pTime, segment(selectedBeat.pIndex), 80, 'mo', 'LineWidth', 2, 'DisplayName', 'P-wave');
end

% Mark Q-wave position
if ~isnan(selectedBeat.qIndex)
    qTime = timeAxis(selectedBeat.qIndex);
    scatter(qTime, segment(selectedBeat.qIndex), 80, 'co', 'LineWidth', 2, 'DisplayName', 'Q-wave');
end

% Mark S-wave position
if ~isnan(selectedBeat.sIndex)
    sTime = timeAxis(selectedBeat.sIndex);
    scatter(sTime, segment(selectedBeat.sIndex), 80, 'go', 'LineWidth', 2, 'DisplayName', 'S-wave');
end

% Mark T-wave position
if ~isnan(selectedBeat.tIndex)
    tTime = timeAxis(selectedBeat.tIndex);
    scatter(tTime, segment(selectedBeat.tIndex), 80, 'ko', 'LineWidth', 2, 'DisplayName', 'T-wave');
end

% Mark P-wave end point
if ~isnan(selectedBeat.pEnd)
    pEndTime = timeAxis(selectedBeat.pEnd);
    scatter(pEndTime, segment(selectedBeat.pEnd), 60, 'mx', 'LineWidth', 2, 'DisplayName', 'P-wave End');
end

% Mark T-wave end point
if ~isnan(selectedBeat.tEnd)
    tEndTime = timeAxis(selectedBeat.tEnd);
    scatter(tEndTime, segment(selectedBeat.tEnd), 60, 'yx', 'LineWidth', 2, 'DisplayName', 'T-wave End');
end

% Get heartbeat type
beatTypeMap = containers.Map([1, 5, 8], {'Normal Beat', 'Premature Ventricular Contraction (PVC)', 'Premature Atrial Contraction (PAC)'});
if isKey(beatTypeMap, selectedBeat.beatType)
    beatTypeName = beatTypeMap(selectedBeat.beatType);
else
    beatTypeName = ['Unknown Type (' num2str(selectedBeat.beatType) ')'];
end

% Add legend and labels
legend('Location', 'best');
title(['Heartbeat Waveform Analysis - ' beatTypeName]);
xlabel('Time (s)');
ylabel('Amplitude (mV)');
grid on;

% Calculate display event range (ensure a certain range before and after R-wave is displayed)
timeRange = 0.4; % Display 0.4 seconds before and after (originally 0.3s, can be adjusted to show a more complete waveform)
xlim([-timeRange, timeRange]);
% ylim([-3, 3]); % Y-axis range can be adjusted according to the actual signal, or automatically adjusted

% % Add heartbeat information text description (commented out part, can be uncommented and translated if needed)
% textStr = {
%     ['Heartbeat Type: ' beatTypeName],
%     ['P-wave Amplitude: ' num2str(getAmplitude(segment, selectedBeat.pIndex), '%.2f') ' mV'],
%     ['QRS Complex Amplitude: ' num2str(getQRSAmplitude(segment, selectedBeat.qIndex, selectedBeat.rIndex, selectedBeat.sIndex), '%.2f') ' mV'],
%     ['T-wave Amplitude: ' num2str(getAmplitude(segment, selectedBeat.tIndex), '%.2f') ' mV'],
%     ['PR Interval: ' num2str(getInterval(selectedBeat.pIndex, selectedBeat.rIndex, fs), '%.2f') ' ms'],
%     ['QRS Interval: ' num2str(getInterval(selectedBeat.qIndex, selectedBeat.sIndex, fs), '%.2f') ' ms'],
%     ['QT Interval: ' num2str(getInterval(selectedBeat.qIndex, selectedBeat.tEnd, fs), '%.2f') ' ms']
% };
% annotation('textbox', [0.01, 0.7, 0.2, 0.25], 'String', textStr, ...
%     'FitBoxToText', 'on', 'BackgroundColor', [1 1 1 0.8], 'EdgeColor', 'k');
% 
% % Output detected waveform information (commented out part)
% fprintf('Heartbeat Index #%d Analysis Result:\n', beatIndex);
% fprintf('Type: %s\n', beatTypeName);
% fprintf('P-wave Detection: %s\n', getDetectionStatus(selectedBeat.pIndex));
% fprintf('Q-wave Detection: %s\n', getDetectionStatus(selectedBeat.qIndex));
% fprintf('R-wave Detection: %s\n', getDetectionStatus(selectedBeat.rIndex));
% fprintf('S-wave Detection: %s\n', getDetectionStatus(selectedBeat.sIndex));
% fprintf('T-wave Detection: %s\n', getDetectionStatus(selectedBeat.tIndex));
end

% Helper function: Get waveform amplitude
function amp = getAmplitude(signal, idx)
    if isnan(idx) || idx <= 0 || idx > length(signal)
        amp = NaN;
    else
        amp = signal(idx);
    end
end

% Helper function: Get QRS complex amplitude
function amp = getQRSAmplitude(signal, qIdx, rIdx, sIdx)
    % If Q-wave and R-wave exist, calculate their amplitude difference
    if ~isnan(qIdx) && ~isnan(rIdx) && qIdx > 0 && rIdx > 0 && qIdx <= length(signal) && rIdx <= length(signal)
        qrAmp = abs(signal(rIdx) - signal(qIdx));
    else
        qrAmp = 0;
    end
    
    % If R-wave and S-wave exist, calculate their amplitude difference
    if ~isnan(rIdx) && ~isnan(sIdx) && rIdx > 0 && sIdx > 0 && rIdx <= length(signal) && sIdx <= length(signal)
        rsAmp = abs(signal(rIdx) - signal(sIdx));
    else
        rsAmp = 0;
    end
    
    % Return the larger amplitude difference
    amp = max(qrAmp, rsAmp);
    
    % If both calculations fail, try using R-wave amplitude directly
    if amp == 0 && ~isnan(rIdx) && rIdx > 0 && rIdx <= length(signal)
        amp = abs(signal(rIdx));
    end
end

% Helper function: Get time interval (milliseconds)
function interval = getInterval(idx1, idx2, fs)
    if isnan(idx1) || isnan(idx2)
        interval = NaN;
    else
        interval = abs(idx2 - idx1) * 1000 / fs; % Convert to milliseconds
    end
end

% Helper function: Get detection status text
function status = getDetectionStatus(idx)
    if isnan(idx)
        status = 'Not Detected';
    else
        status = 'Detected';
    end
end