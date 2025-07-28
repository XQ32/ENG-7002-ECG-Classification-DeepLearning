function displayBeatStatistics(allBeatInfo)
%displayBeatStatistics Display statistical information of final heartbeat counts by type
%   This function counts and prints the number and percentage of each heartbeat type
%   based on the allBeatInfo.originalAnnotationChar field.
%
%   Input:
%       allBeatInfo - Structure array containing all heartbeat information

actual_beat_count = length(allBeatInfo);

if actual_beat_count > 0
    fprintf('\n\n--- Final Heartbeat Count Statistics (based on allBeatInfo.originalAnnotationChar) ---\n');

    all_original_chars_final = arrayfun(@(x) x.originalAnnotationChar, allBeatInfo, 'UniformOutput', false);

    unique_char_types_final = unique(all_original_chars_final);
    total_beats_in_allBeatInfo = actual_beat_count;
    fprintf('Total heartbeats processed in allBeatInfo: %d\n', total_beats_in_allBeatInfo);

    if total_beats_in_allBeatInfo > 0
        for char_idx = 1:length(unique_char_types_final)
            current_char = unique_char_types_final{char_idx};
            if isempty(current_char)
                count = sum(cellfun('isempty', all_original_chars_final));
                 fprintf('  Type (empty character): %d beats (%.2f%%)\n', count, (count/total_beats_in_allBeatInfo)*100);
            else
                count = sum(strcmp(all_original_chars_final, current_char));
                fprintf('  Type ''%s'': %d beats (%.2f%%)\n', current_char, count, (count/total_beats_in_allBeatInfo)*100);
            end
        end
    end
else
    fprintf('\n\nallBeatInfo is empty, no heartbeat data available for statistics.\n');
end

end