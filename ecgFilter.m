function [ecg_filtered, filter_info] = ecgFilter(ecg, fs, method_index, power_line_freq)
% ecgFilter - Filters the ECG signal using various filtering methods
% 
% Inputs:
%   ecg - Original ECG signal (can be a row or column vector)
%   fs - Sampling frequency (Hz)
%   method_index - Index of the filtering method to use
%                  1 - Original FIR high-pass filter
%                  2 - Butterworth IIR filter method
%   power_line_freq - Power line interference frequency, can be 50Hz or 60Hz (default: 60Hz)
% 
% Outputs:
%   ecg_filtered - ECG signal filtered according to the selected method (column vector)
%   filter_info - Struct containing information related to the filtering
%
% This function supports two methods for removing baseline wander and includes a low-pass filter to remove high-frequency noise.

% Ensure ecg is a column vector to maintain consistency in processing
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
    warning('Power line frequency %d Hz is invalid, only 50Hz or 60Hz are supported. Defaulting to 60Hz.', power_line_freq);
    power_line_freq = 60;
end

% Cutoff frequency set to 0.5Hz
cutoff_freq = 0.5;

% Create output information struct
filter_info = struct();
filter_info.method_index = method_index;
filter_info.cutoff_freq = cutoff_freq;
filter_info.sampling_freq = fs;
filter_info.power_line_freq = power_line_freq;
filter_info.original = ecg;

%% Step 1: Use a notch filter to eliminate power line interference (50Hz or 60Hz)
% Design a narrow-band power line notch filter
wo = power_line_freq/(fs/2);  % Normalized cutoff frequency
bw = wo/35;      % Set a narrow bandwidth (Q factor = 35)
[b, a] = iirnotch(wo, bw);  % Design an IIR notch filter

% Apply the notch filter using zero-phase filtering
ecg_notch_filtered = filtfilt(b, a, ecg);
filter_info.notch_filtered = ecg_notch_filtered;

% Add filtering for power line harmonics (e.g., 2x/3x the power line frequency)
if fs > power_line_freq * 3 % Ensure the sampling rate is high enough to filter harmonics
    % Add a 2x power line frequency filter
    wo_2nd = (power_line_freq*2)/(fs/2);
    if wo_2nd < 0.95 % Ensure normalized frequency is less than 1
        bw_2nd = wo_2nd/40; % Can make the harmonic filter bandwidth narrower
        [b_2nd, a_2nd] = iirnotch(wo_2nd, bw_2nd);
        ecg_notch_filtered = filtfilt(b_2nd, a_2nd, ecg_notch_filtered);
        filter_info.has_2nd_harmonic_filter = true;
    end
end

%% Step 2: Select a baseline wander removal method based on method_index
switch method_index
    case 1
        % Method 1: Original FIR high-pass filter
        [ecg_baseline_filtered, filter_params] = apply_FIR_filter(ecg_notch_filtered, fs, cutoff_freq);
        filter_info.name = 'Original FIR High-pass Filter';
        filter_info.filter_params = filter_params;
        
    case 2
        % Method 2: Butterworth IIR filter method
        try
            [ecg_baseline_filtered, filter_params] = apply_IIR_filter(ecg_notch_filtered, fs, cutoff_freq);
            filter_info.name = 'Butterworth IIR Filter';
            filter_info.filter_params = filter_params;
        catch e
            warning('Error during execution of Method 2 (IIR): %s, using Method 1 as a fallback', e.message);
            [ecg_baseline_filtered, filter_params] = apply_FIR_filter(ecg_notch_filtered, fs, cutoff_freq);
            filter_info.name = 'Original FIR High-pass Filter (Fallback)';
            filter_info.filter_params = filter_params;
            filter_info.error = e.message;
        end
        
    otherwise
        % Default to Method 1
        warning('Invalid method index: %d, will use Method 1', method_index);
        [ecg_baseline_filtered, filter_params] = apply_FIR_filter(ecg_notch_filtered, fs, cutoff_freq);
        filter_info.name = 'Original FIR High-pass Filter (Default)';
        filter_info.filter_params = filter_params;
end

%% Step 3: Apply a low-pass filter to remove high-frequency noise
% Low-pass filter cutoff frequency is set to 150Hz to remove high-frequency noise while preserving the main ECG components
lowpass_cutoff = 150; % Hz

% Design a low-pass Butterworth filter
if fs > 2 * lowpass_cutoff % Ensure the sampling rate meets the Nyquist theorem
    [b_lp, a_lp] = butter(4, lowpass_cutoff/(fs/2), 'low');
    ecg_filtered = filtfilt(b_lp, a_lp, ecg_baseline_filtered);
    
    % Record low-pass filter information
    filter_info.lowpass_applied = true;
    filter_info.lowpass_cutoff = lowpass_cutoff;
    filter_info.lowpass_params.order = 4;
    filter_info.lowpass_params.b = b_lp;
    filter_info.lowpass_params.a = a_lp;
    
    fprintf('  Applied %d Hz low-pass filter to remove high-frequency noise\n', lowpass_cutoff);
else
    % If the sampling rate is too low, skip low-pass filtering
    ecg_filtered = ecg_baseline_filtered;
    filter_info.lowpass_applied = false;
    warning('Sampling rate %d Hz is too low, skipping %d Hz low-pass filtering', fs, lowpass_cutoff);
end

end

% Method 1: Original FIR high-pass filter function
function [filtered_signal, params] = apply_FIR_filter(signal, fs, cutoff_freq)
    % Get signal length
    signal_length = length(signal);
    
    % Calculate the maximum acceptable filter order (1/10 of the signal length)
    max_filter_order = floor(signal_length/10);
    
    fcuts = [(cutoff_freq-0.07) (cutoff_freq)];
    mags = [0 1];
    % Increase error tolerance to reduce the required filter order
    devs = [0.01 0.005];  % Relax error requirements
    
    % Design the filter using kaiserord
    [n, Wn, beta, ftype] = kaiserord(fcuts, mags, devs, fs);
    
    % Limit the filter order to avoid excessively high orders
    if n > max_filter_order
        % If the calculated order is too high, use a fixed lower order
        original_n = n;
        n = max_filter_order;
        fprintf('Filter order has been reduced from %d to %d\n', original_n, max_filter_order);
    end
    
    % Design the FIR filter
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