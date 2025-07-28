function allBeatInfo = cleanHeartbeatData(allBeatInfo)
%cleanHeartbeatData Clean heartbeat data records containing NaN values
%   This function checks specified heartbeat feature points, removes any records containing NaN values,
%   and returns a cleaned structure array.
%
%   Input:
%       allBeatInfo - Structure array containing all detected heartbeat information
%
%   Output:
%       allBeatInfo - Cleaned structure array

fprintf('\nStarting to clean heartbeats containing NaN values in allBeatInfo...\n');

original_beat_count = length(allBeatInfo);
fprintf('Total %d heartbeat records before cleaning\n', original_beat_count);

if original_beat_count == 0
    fprintf('Data is empty, no cleaning needed.\n');
    return;
end

fprintf('Using memory-optimized NaN detection method...\n');

has_nan = false(original_beat_count, 1);
nan_counts = zeros(6, 1);  % [pIndex, qIndex, sIndex, tIndex, pEnd, tEnd]

for i = 1:original_beat_count
    current_beat = allBeatInfo(i);

    is_nan_p = isnan(current_beat.pIndex);
    is_nan_q = isnan(current_beat.qIndex);
    is_nan_s = isnan(current_beat.sIndex);
    is_nan_t = isnan(current_beat.tIndex);
    is_nan_pEnd = isnan(current_beat.pEnd);
    is_nan_tEnd = isnan(current_beat.tEnd);

    nan_counts(1) = nan_counts(1) + is_nan_p;
    nan_counts(2) = nan_counts(2) + is_nan_q;
    nan_counts(3) = nan_counts(3) + is_nan_s;
    nan_counts(4) = nan_counts(4) + is_nan_t;
    nan_counts(5) = nan_counts(5) + is_nan_pEnd;
    nan_counts(6) = nan_counts(6) + is_nan_tEnd;

    if is_nan_p || is_nan_q || is_nan_s || is_nan_t || is_nan_pEnd || is_nan_tEnd
        has_nan(i) = true;
    end
end

fprintf('Statistics of fields containing NaN values:\n');
fprintf('  P-wave index (pIndex): %d heartbeats\n', nan_counts(1));
fprintf('  Q-wave index (qIndex): %d heartbeats\n', nan_counts(2));
fprintf('  S-wave index (sIndex): %d heartbeats\n', nan_counts(3));
fprintf('  T-wave index (tIndex): %d heartbeats\n', nan_counts(4));
fprintf('  P-wave end point (pEnd): %d heartbeats\n', nan_counts(5));
fprintf('  T-wave end point (tEnd): %d heartbeats\n', nan_counts(6));

invalid_count = sum(has_nan);

if invalid_count > 0
    allBeatInfo = allBeatInfo(~has_nan);
    fprintf('Removed %d heartbeat records containing NaN values (%.2f%% of total)\n', invalid_count, (invalid_count/original_beat_count)*100);
else
    fprintf('No NaN records found, no cleaning needed\n');
end

fprintf('Remaining %d heartbeat records after cleaning\n', length(allBeatInfo));

end