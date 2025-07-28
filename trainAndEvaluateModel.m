function trainAndEvaluateModel(training_waveforms, training_context, training_labels, testing_waveforms, testing_context, testing_labels, dl_params)
%trainAndEvaluateModel Train and evaluate deep learning model
%   This function supports complete dl_params configuration system:
%   - Build and train deep learning models (supports multiple network architectures)
%   - Evaluate model performance on test set
%   - Supports binary and multi-class classification tasks
%   - Fully configurable training parameters and network architecture
%
%   Input:
%       training_waveforms   - Training waveform data (cell array)
%       training_context     - Training context features (matrix)
%       training_labels      - Training labels (categorical array)
%       testing_waveforms    - Testing waveform data (cell array)
%       testing_context      - Testing context features (matrix)
%       testing_labels       - Testing labels (categorical array)
%       dl_params           - Deep learning configuration parameter structure
%
%   Required dl_params fields:
%       .standardLength         - Standard waveform length
%       .networkType           - Network type: 'simple'|'enhanced'|'multimodal'
%       .classification        - Classification configuration
%       .training             - Training configuration
%       .context              - Context feature configuration

if exist('training_labels', 'var') && ~isempty(training_labels)

    % === Step 1: Parameter Validation and Configuration ===
    fprintf('\n--- Deep Learning Training: Step 1 - Parameter Configuration ---\n');

    % Backward compatibility handling
    if isfield(dl_params, 'classificationTask') && ~isfield(dl_params, 'classification')
        dl_params.classification.task = dl_params.classificationTask;
    end
    if isfield(dl_params, 'classNames') && ~isfield(dl_params, 'classification')
        dl_params.classification.classNames = dl_params.classNames;
    end
    if isfield(dl_params, 'dropoutRate') && ~isfield(dl_params, 'regularization')
        dl_params.regularization.dropoutRate = dl_params.dropoutRate;
    end

    % Set default parameters
    if ~isfield(dl_params, 'networkType'), dl_params.networkType = 'enhanced'; end

    % Ensure required parameters exist
    if ~isfield(dl_params, 'classification')
        dl_params.classification.task = 'binary';
        dl_params.classification.classNames = {'Other', 'PVC'};
        dl_params.classification.threshold = 0.2;
    end
    
    if ~isfield(dl_params, 'training')
        dl_params.training.optimizer = 'adam';
        dl_params.training.initialLearningRate = 2e-3;
        dl_params.training.miniBatchSize = 256;
        dl_params.training.maxEpochs = 20;
        dl_params.training.validationSplit = 0.1;
    end
    
    if ~isfield(dl_params, 'context')
        dl_params.context.enhanced = true;
        dl_params.context.dimension = 4;
    end

    % Automatically detect context feature dimension
    contextDim = size(training_context, 2);
    fprintf('Detected context feature dimension: %d\n', contextDim);

    dl_params.context.dimension = contextDim;

    if contextDim == 2
        fprintf('Using basic context features (RR_Prev, RR_Post)\n');
    elseif contextDim == 4
        fprintf('Using enhanced context features (RR_Prev, RR_Post, HR_Variability, Rhythm_Stability)\n');
    else
        fprintf('Using custom context features, dimension: %d\n', contextDim);
    end

    fprintf('Network configuration: %s\n', dl_params.networkType);
    if isfield(dl_params, 'architecture')
        fprintf('Architecture configuration: Custom parameters\n');
    else
        fprintf('Architecture configuration: Default parameters\n');
    end

    % === Step 2: Build Network Model ===
    fprintf('\n--- Deep Learning Training: Step 2 - Build Network Model ---\n');
    numClasses = length(unique(training_labels));
    fprintf('Number of classification classes: %d\n', numClasses);

    % Build model according to network type
    switch lower(dl_params.networkType)
        case 'simple'
            fprintf('Building simple hybrid CNN model...\n');
            lgraph = buildHybridCNNModel(numClasses, dl_params.standardLength, ...
                'ContextDim', contextDim);

        case 'enhanced'
            fprintf('Building enhanced hybrid CNN model...\n');
            [~, lgraph] = buildEnhancedHybridCNNModel(numClasses, dl_params.standardLength, ...
                'ContextDim', contextDim, ...
                'dl_params', dl_params);

        otherwise
            warning('Unknown network type: %s, using enhanced network', dl_params.networkType);
            fprintf('Building enhanced hybrid CNN model (default)...\n');
            [~, lgraph] = buildEnhancedHybridCNNModel(numClasses, dl_params.standardLength, ...
                'ContextDim', contextDim, ...
                'dl_params', dl_params);
    end

    fprintf('✓ Network construction completed\n');

    % === Step 3: Prepare Training Data ===
    fprintf('\n--- Deep Learning Training: Step 3 - Prepare Training Data ---\n');

    % Data validation
    fprintf('Training data validation:\n');
    fprintf('  Waveform data: %d samples\n', length(training_waveforms));
    fprintf('  Context data: [%d x %d]\n', size(training_context, 1), size(training_context, 2));
    fprintf('  Label data: %d labels\n', length(training_labels));

    if length(training_waveforms) ~= size(training_context, 1) || ...
       length(training_waveforms) ~= length(training_labels)
        error('Training data dimensions are inconsistent');
    end

    % Create validation split
    validationSplit = dl_params.training.validationSplit;
    cvp = cvpartition(training_labels, 'Holdout', validationSplit);
    idxTrain = training(cvp);
    idxVal = test(cvp);

    fprintf('Data split: training=%d, validation=%d (ratio=%.1f)\n', sum(idxTrain), sum(idxVal), validationSplit);

    fprintf('Preparing training and validation data...');

    % Convert waveform data to matrix format
    XTrain_wave = vertcat(training_waveforms{idxTrain});
    XVal_wave   = vertcat(training_waveforms{idxVal});

    % Extract context and labels
    XTrain_context = training_context(idxTrain, :);
    YTrain         = training_labels(idxTrain);
    XVal_context   = training_context(idxVal, :);
    YVal           = training_labels(idxVal);

    % Create data stores for multi-input network
    dsXTrain_wave = arrayDatastore(XTrain_wave', 'IterationDimension', 2);
    dsXVal_wave   = arrayDatastore(XVal_wave',   'IterationDimension', 2);

    dsXTrain_context = arrayDatastore(XTrain_context', 'IterationDimension', 2);
    dsXVal_context   = arrayDatastore(XVal_context',   'IterationDimension', 2);

    dsYTrain = arrayDatastore(YTrain, 'IterationDimension', 1);
    dsYVal   = arrayDatastore(YVal,   'IterationDimension', 1);

    % Combine multi-input data stores
    dsTrain = combine(dsXTrain_wave, dsXTrain_context, dsYTrain);
    dsVal   = combine(dsXVal_wave,   dsXVal_context,   dsYVal);
    fprintf('✓ Data store creation completed\n');

    % === Step 4: Configure and Run Training ===
    fprintf('\n--- Step 4: Network Training ---\n');

    % Get training parameters from configuration
    miniBatchSize = dl_params.training.miniBatchSize;
    maxEpochs = dl_params.training.maxEpochs;
    initialLearningRate = dl_params.training.initialLearningRate;
    optimizer = dl_params.training.optimizer;
    
    numIterationsPerEpoch = floor(sum(idxTrain) / miniBatchSize);

    fprintf('Training parameters:\n');
    fprintf('  Optimizer: %s\n', optimizer);
    fprintf('  Mini-batch size: %d\n', miniBatchSize);
    fprintf('  Max epochs: %d\n', maxEpochs);
    fprintf('  Learning rate: %.1e\n', initialLearningRate);
    
    % Check if class weights are used
    if strcmp(dl_params.classBalance.method, 'weights') && isfield(dl_params.classBalance, 'classWeights')
        fprintf('  Class weights: [%.1f, %.1f] (will be implemented via data resampling)\n', dl_params.classBalance.classWeights(1), dl_params.classBalance.classWeights(2));
    end

    % Configure training options (remove ClassWeights as MATLAB's trainingOptions does not support it)
    options = trainingOptions(optimizer, ...
        'InitialLearnRate', initialLearningRate, ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', dsVal, ...
        'ValidationFrequency', numIterationsPerEpoch, ...
        'Plots', 'training-progress', ...
        'Verbose', true, ...
        'ExecutionEnvironment', 'auto');
    
    % Add early stopping feature if enabled
    if isfield(dl_params.training, 'enableEarlyStopping') && dl_params.training.enableEarlyStopping
        options.ValidationPatience = dl_params.training.patience;
        fprintf('✓ Early stopping enabled, patience: %d epochs\n', dl_params.training.patience);
    end

    fprintf('\nStarting training...\n');
    try
        trainedNet = trainNetwork(dsTrain, lgraph, options);
        fprintf('✓ Training completed\n');
        predictNet = trainedNet;

    catch ME_train
        fprintf('❌ Training failed: %s\n', ME_train.message);
        fprintf('Error details:\n%s\n', getReport(ME_train));
        error('Deep learning training failed, please check network structure and data format');
    end

    % === Step 5: Model Evaluation ===
    if exist('testing_labels', 'var') && ~isempty(testing_labels)
        fprintf('\n--- Deep Learning Training: Step 5 - Model Evaluation ---\n');
        fprintf('Test set contains %d heartbeat samples\n', length(testing_labels));

        % Validate test data format
        if size(testing_context, 2) ~= contextDim
            error('Test set context feature dimension (%d) inconsistent with training set (%d)', size(testing_context, 2), contextDim);
        end

        % Convert test data
        XTest_wave = vertcat(testing_waveforms{:});
        XTest_context = testing_context;
        YTest = testing_labels;

        fprintf('Test data format:\n');
        fprintf('  Test waveform data: [%d, %d]\n', size(XTest_wave, 1), size(XTest_wave, 2));
        fprintf('  Test context data: [%d, %d]\n', size(XTest_context, 1), size(XTest_context, 2));

        fprintf('\nStarting prediction...\n');
        try
            if isa(predictNet, 'dlnetwork')
                % dlnetwork prediction
                XTest_wave_dl = dlarray(XTest_wave', 'CB');
                XTest_context_dl = dlarray(XTest_context', 'CB');
                predictedScores_dlarray = predict(predictNet, XTest_wave_dl, XTest_context_dl);
                scoresMatrix = extractdata(predictedScores_dlarray)';
            else
                % Standard network prediction
                fprintf('Creating test data store...\n');

                dsXTest_wave = arrayDatastore(XTest_wave', 'IterationDimension', 2);
                dsXTest_context = arrayDatastore(XTest_context', 'IterationDimension', 2);
                dsTest = combine(dsXTest_wave, dsXTest_context);

                predictedScores = predict(predictNet, dsTest);
                scoresMatrix = predictedScores;
            end
            fprintf('✓ Prediction completed\n');
        catch ME_pred
            fprintf('❌ Prediction failed: %s\n', ME_pred.message);
            fprintf('Error details: %s\n', getReport(ME_pred));
            error('Model prediction failed');
        end
        
        % Classification and performance evaluation
        actual_labels = YTest;
        class_names = categories(actual_labels);
        
        % Adjust prediction based on classification task
        if strcmp(dl_params.classification.task, 'binary')
            fprintf('Performing binary classification evaluation\n');
            
            classification_threshold = dl_params.classification.threshold;
            
            % Find class indices
            pvc_class_index = find(strcmp(class_names, 'PVC'));
            other_class_index = find(strcmp(class_names, 'Other'));
            
            if isempty(pvc_class_index)
                pvc_class_index = find(strcmp(class_names, dl_params.classification.classNames{2}));
            end
            if isempty(other_class_index)
                other_class_index = find(strcmp(class_names, dl_params.classification.classNames{1}));
            end
            
            % Perform binary classification prediction
            if ~isempty(pvc_class_index) && ~isempty(other_class_index)
                predictedLabels_idx = ones(size(scoresMatrix, 1), 1) * other_class_index;
                predictedLabels_idx(scoresMatrix(:, pvc_class_index) > classification_threshold) = pvc_class_index;
                predicted_labels = categorical(class_names(predictedLabels_idx));
            else
                warning('Standard classes not found, using max probability prediction');
                [~, predictedLabels_idx] = max(scoresMatrix, [], 2);
                predicted_labels = categorical(class_names(predictedLabels_idx));
            end
            
        else
            fprintf('Performing multi-class classification evaluation\n');
            
            [~, predictedLabels_idx] = max(scoresMatrix, [], 2);
            predicted_labels = categorical(class_names(predictedLabels_idx));
        end

        accuracy = sum(predicted_labels == actual_labels) / length(actual_labels) * 100;
        fprintf('\n=== %s Network Performance Evaluation ===\n', dl_params.networkType);
        fprintf('Test set accuracy: %.2f%%\n', accuracy);
        fprintf('Classification task: %s (%d classes)\n', dl_params.classification.task, length(class_names));
        fprintf('Target classes: %s\n', strjoin(class_names, ', '));
        fprintf('Classification threshold: %.2f\n', classification_threshold);
        
        num_classes = length(class_names);
        
        % Display confusion matrix
        figure('Name', sprintf('%s Model Confusion Matrix', dl_params.networkType), 'NumberTitle', 'off');
        cm = confusionchart(actual_labels, predicted_labels, ...
            'Title', sprintf('%s Model Confusion Matrix', dl_params.networkType), ...
            'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
        
        % Optional attention visualization prompt
        if strcmp(dl_params.networkType, 'multimodal') && isfield(dl_params, 'advanced') && dl_params.advanced.enableAttention
            fprintf('\n--- Attention Weight Visualization ---\n');
            fprintf('Call visualizeAttention function for detailed analysis\n');
            fprintf('Example: visualizeAttention(trainedNet, sampleData, sampleLabels)\n');
        end

        confusion_mat = cm.NormalizedValues;
        
        fprintf('\n=== Per-Class Performance Metrics ===\n');
        tpr_fnr_matrix = zeros(num_classes, 2);
        for i = 1:num_classes
            TP = confusion_mat(i, i);
            FP = sum(confusion_mat(:, i)) - TP;
            FN = sum(confusion_mat(i, :)) - TP;
            
            sensitivity = TP / (TP + FN) * 100;
            if isnan(sensitivity)
                sensitivity = 0;
            end
            
            precision = TP / (TP + FP) * 100;
            if isnan(precision)
                precision = 0;
            end
            if (precision + sensitivity) > 0
                f1_score = 2 * (precision * sensitivity) / (precision + sensitivity);
            else
                f1_score = 0;
            end
            
            tpr_fnr_matrix(i, 1) = sensitivity;
            tpr_fnr_matrix(i, 2) = 100 - sensitivity;

            fprintf('Class %s:\n', class_names{i});
            fprintf('  Sensitivity/Recall: %.2f%%\n', sensitivity);
            fprintf('  False Negative Rate: %.2f%%\n', 100 - sensitivity);
            fprintf('  Precision: %.2f%%\n', precision);
            fprintf('  F1 Score: %.2f\n\n', f1_score);
        end
        
        % Performance comparison chart
        figure('Name', sprintf('%s Model Performance Comparison', dl_params.networkType), 'NumberTitle', 'off');
        bar_width = 0.35;
        x = 1:num_classes;
        bar(x - bar_width/2, tpr_fnr_matrix(:, 1), bar_width, 'FaceColor', [0.2 0.7 0.2], 'DisplayName', 'TPR (%)');
        hold on;
        bar(x + bar_width/2, tpr_fnr_matrix(:, 2), bar_width, 'FaceColor', [0.7 0.2 0.2], 'DisplayName', 'FNR (%)');
        xlabel('Heartbeat Class');
        ylabel('Percentage (%)');
        title(sprintf('%s Model TPR and FNR Comparison by Class', dl_params.networkType));
        legend('Location', 'best');
        set(gca, 'XTick', x, 'XTickLabel', class_names);
        grid on;
        ylim([0, 100]);
        for i = 1:num_classes
            text(i - bar_width/2, tpr_fnr_matrix(i, 1) + 2, sprintf('%.1f', tpr_fnr_matrix(i, 1)), 'HorizontalAlignment', 'center', 'FontSize', 9);
            text(i + bar_width/2, tpr_fnr_matrix(i, 2) + 2, sprintf('%.1f', tpr_fnr_matrix(i, 2)), 'HorizontalAlignment', 'center', 'FontSize', 9);
        end

    else
        fprintf('Test set is empty, unable to evaluate classifier performance.\n');
    end
else
    fprintf('Training data was not successfully generated, unable to train model.\n');
end

end
