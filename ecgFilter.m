function [ecg_filtered, filter_info] = ecgFilter(ecg, fs, method_index, power_line_freq)
% ecgFilter - Filters ECG signal using various methods
% 
% Input:
%   ecg - Raw ECG signal (can be row or column vector)
%   fs - Sampling frequency (Hz)
%   method_index - Index of the filtering method to use
%                  1 - Original FIR high-pass filter
%                  2 - Butterworth IIR filter method
%   power_line_freq - Power line interference frequency, can be 50Hz or 60Hz (default: 60Hz)
% 
% Output:
%   ecg_filtered - ECG signal filtered according to the selected method (column vector)
%   filter_info - Structure containing filter-related information
%
% This function supports two methods for removing baseline wander

% Ensure ecg is a column vector for consistent processing
if size(ecg, 2) > size(ecg, 1)
    ecg = ecg';
end

% Default parameter settings
if nargin < 3
    method_index = 1;
end
if nargin < 4
    power_line_freq = 60; % Default 60Hz power line interference
end

% Validate power line interference parameter
if power_line_freq ~= 50 && power_line_freq ~= 60
    warning('Invalid power line frequency %d Hz. Only 50Hz or 60Hz are supported. Using default 60Hz.', power_line_freq);
    power_line_freq = 60;
end

% Cutoff frequency set to 0.5Hz
cutoff_freq = 0.5;

% Create output information structure
filter_info = struct();
filter_info.method_index = method_index;
filter_info.cutoff_freq = cutoff_freq;
filter_info.sampling_freq = fs;
filter_info.power_line_freq = power_line_freq;
filter_info.original = ecg;

%% Step 1: Use notch filter to eliminate power line interference (50Hz or 60Hz)
% Design a narrow-band power line notch filter
wo = power_line_freq/(fs/2);  % Normalized cutoff frequency
bw = wo/35;      % Set narrow bandwidth (Q factor = 35)
[b, a] = iirnotch(wo, bw);  % Design IIR notch filter

% Apply notch filter using zero-phase filtering
ecg_notch_filtered = filtfilt(b, a, ecg);
filter_info.notch_filtered = ecg_notch_filtered;

% Add power line harmonic filtering (e.g., 2x/3x power line frequency)
if fs > power_line_freq * 3 % Ensure sampling rate is high enough to filter harmonics
    % Add 2x power line frequency filter
    wo_2nd = (power_line_freq*2)/(fs/2);
    if wo_2nd < 0.95 % Ensure normalized frequency is less than 1
        bw_2nd = wo_2nd/40; % Can make harmonic filter bandwidth narrower
        [b_2nd, a_2nd] = iirnotch(wo_2nd, bw_2nd);
        ecg_notch_filtered = filtfilt(b_2nd, a_2nd, ecg_notch_filtered);
        filter_info.has_2nd_harmonic_filter = true;
    end
end

%% Step 2: Select baseline wander removal method based on method_index
switch method_index
    case 1
        % Method 1: Original FIR high-pass filter
        [ecg_filtered, filter_params] = apply_FIR_filter(ecg_notch_filtered, fs, cutoff_freq);
        filter_info.name = 'Original FIR High-pass Filter';
        filter_info.filter_params = filter_params;
        
    case 2
        % Method 2: Butterworth IIR filter method
        try
            [ecg_filtered, filter_params] = apply_IIR_filter(ecg_notch_filtered, fs, cutoff_freq);
            filter_info.name = 'Butterworth IIR Filter';
            filter_info.filter_params = filter_params;
        catch e
            warning('Error during Method 2 (IIR) execution: %s. Using Method 1 as fallback.', e.message);
            [ecg_filtered, filter_params] = apply_FIR_filter(ecg_notch_filtered, fs, cutoff_freq);
            filter_info.name = 'Original FIR High-pass Filter (Fallback)';
            filter_info.filter_params = filter_params;
            filter_info.error = e.message;
        end
        
    otherwise
        % Default to Method 1
        warning('Invalid method index: %d. Using Method 1.', method_index);
        [ecg_filtered, filter_params] = apply_FIR_filter(ecg_notch_filtered, fs, cutoff_freq);
        filter_info.name = 'Original FIR High-pass Filter (Default)';
        filter_info.filter_params = filter_params;
end

end

% Method 1: Original FIR high-pass filter function
function [filtered_signal, params] = apply_FIR_filter(signal, fs, cutoff_freq)
    % Get signal length
    signal_length = length(signal);
    
    % Calculate acceptable maximum filter order (1/10 of signal length)
    max_filter_order = floor(signal_length/10);
    
    fcuts = [(cutoff_freq-0.07) (cutoff_freq)];
    mags = [0 1];
    % Increase error tolerance to reduce required filter order
    devs = [0.01 0.005];  % Relax error requirements
    
    % Design filter using kaiserord
    [n, Wn, beta, ftype] = kaiserord(fcuts, mags, devs, fs);
    
    % Limit filter order to avoid excessively high order
    if n > max_filter_order
        % If calculated order is too high, use a fixed lower order
        original_n = n;
        n = max_filter_order;
        fprintf('Filter order reduced from %d to %d\n', original_n, max_filter_order);
    end
    
    % Design FIR filter
    b = fir1(n, Wn, ftype, kaiser(n+1, beta), 'noscale');
    a = 1;
    
    % Apply zero-phase filtering
    filtered_signal = filtfilt(b, a, signal);
    
    % Record parameters
    params.filter_order = n;
    params.window = 'kaiser';
    params.beta = beta;
    params.cutoff = Wn;
    params.filter_type = ftype;
    params.b = b;
    params.a = a;
end

% Method 2: Butterworth IIR filter function
function [filtered_signal, params] = apply_IIR_filter(signal, fs, cutoff_freq)
    % Design Butterworth IIR filter
    order = 4;  % Filter order
    [b, a] = butter(order, cutoff_freq/(fs/2), 'high');
    
    % Apply zero-phase filtering
    filtered_signal = filtfilt(b, a, signal);
    
    % Record parameters
    params.filter_order = order;
    params.filter_type = 'butterworth';
    params.cutoff = cutoff_freq;
    params.b = b;
    params.a = a;
end