clear all
close all

CDir = cd;
fileNames = dir('*.mat');
cd(CDir);

predictions = zeros(1, length(fileNames));
target = textread('targetData.txt', '%f', 'delimiter', '\n');
transpose(target);

for i = 1:length(fileNames)
    data = load(fileNames(i).name);
    data = data.data;

    % smooth and denoises the data with preprocess()
    col1 = preprocess(data(:,1));
    col2 = preprocess(data(:,2));
    col3 = preprocess(data(:,3));

    % note: diff = derivative
    firstDiff1 = diff(col1);
    firstDiff2 = diff(col2);
    firstDiff3 = diff(col3);
    diffs1 = diff(col1).^2;
    diffs2 = diff(col2).^2;
    diffs3 = diff(col3).^2;
    smoothDiffs1 = smoothdata(diffs1,'gaussian','SmoothingFactor',0.25);
    smoothDiffs2 = smoothdata(diffs2,'gaussian','SmoothingFactor',0.25);
    smoothDiffs3 = smoothdata(diffs3,'gaussian','SmoothingFactor',0.25);

    % variances
    var1 = var(smoothDiffs1);
    var2 = var(smoothDiffs2);
    var3 = var(smoothDiffs3);

    % standard deviations
    sig1 = sqrt(var1);
    sig2 = sqrt(var2);
    sig3 = sqrt(var3);

    properBleaches = 0;
    col1Bleaches = 0;
    col2Bleaches = 0;
    col3Bleaches = 0;
    
    peaks1 = findpeaks(smoothDiffs1, 'MinPeakDistance', 50);
    peaks2 = findpeaks(smoothDiffs2, 'MinPeakDistance', 50);
    peaks3 = findpeaks(smoothDiffs3, 'MinPeakDistance', 50);

    % identifies and stores the drop/spike locations
    for j = 1:length(peaks1)
        if peaks1(j) >= max(smoothDiffs1) * 0.25 && firstDiff1(j) > 0 % mean(smoothDiffs1) + 8 * sig1
            dipLoc = find(smoothDiffs1 == peaks1(j));
            % disp(dipLoc);
            if truePhotobleach(dipLoc, smoothDiffs2, smoothDiffs3, peaks2, peaks3, firstDiff2, firstDiff3)
                properBleaches = properBleaches + 1;
            end
        end
    end

    % identifies the number of drops/spikes in each fluorophore trace
    for j = 1:length(peaks1)
        if peaks1(j) >= max(smoothDiffs1) * 0.25 % mean(smoothDiffs1) + 8 * sig1
            col1Bleaches = col1Bleaches + 1;
        end
    end
    for j = 1:length(peaks2)
        if peaks2(j) >= max(smoothDiffs2) * 0.25 % mean(smoothDiffs2) + 8 * sig2 
            col2Bleaches = col2Bleaches + 1;
        end
    end
    for j = 1:length(peaks3)
        if peaks3(j) >= max(smoothDiffs3) * 0.25 % mean(smoothDiffs3) + 8 * sig3
            col3Bleaches = col3Bleaches + 1;
        end
    end

    % check to see if variances are too high
    if var(smoothDiffs1) < 10000000
        properBleaches = 0;
        col1Bleaches = 0;
    end
    if var(smoothDiffs2) < 10000
        properBleaches = 0;
        col2Bleaches = 0;
    end
    if var(smoothDiffs3) < 10000000
        properBleaches = 0;
        col3Bleaches = 0;
    end

    % manual classification system based on number of drops in each fluorophore trace
    % predictions from this are referred to as benchmark model
    if col1Bleaches == 0 && col2Bleaches == 0 && col3Bleaches == 0
        predictions(i) = 0;
    elseif col1Bleaches > 1 || col2Bleaches > 1 || col3Bleaches > 1 % properBleaches ~= 1
        predictions(i) = 0;
    else
        predictions(i) = 1;
    end

end

TP = 0;
FN = 0;
FP = 0;
TN = 0;

% calculates values for a confusion matrix - helps to calculate accuracy
for i = 1:length(fileNames)
    if predictions(i) == target(i)
        if predictions(i) == 1
            TP = TP + 1;
        else
            TN = TN + 1;
        end
    else
        if predictions(i) == 1
            %disp(i);
            FP = FP + 1;
        else
            disp(i);
            FN = FN + 1;
        end
    end
end

fprintf('[TP, FN, FP, TN] = [%d, %d, %d, %d]\n', TP, FN, FP, TN);
accuracy = 100 * (TP + TN) / (TP + TN + FP + FN);
fprintf('Accuracy: %f\n', accuracy);

function y = preprocess(x)
    %  Preprocess (smooth and denoise) input x
    %  This function expects an input vector x.

    denoised = wdenoise(x, 8, ...
        'Wavelet', 'sym8', ...
        'DenoisingMethod', 'Bayes', ...
        'ThresholdRule', 'Soft', ...
        'NoiseEstimate', 'LevelIndependent');

    y = smoothdata(denoised,'gaussian','SmoothingFactor',0.07);
end





function bleach = truePhotobleach(a, smoothDiffsAcc, smoothDiffsAlex, peaksAcc, peaksAlex, firstDiffAcc, firstDiffAlex)
    bleach = false;

    for i = 0:10
        if a - i > 0 && (ismember(smoothDiffsAcc(a - i), peaksAcc) || ismember(smoothDiffsAlex(a - i), peaksAlex)) && (firstDiffAcc(a - i) < 0 || firstDiffAlex(a - i) < 0)
            if((smoothDiffsAcc(a - i) >= max(smoothDiffsAcc) * 0.25) || (smoothDiffsAlex(a - i) >= max(smoothDiffsAlex) * 0.25))
                bleach = true;
            end
        end
        if a + i <= length(smoothDiffsAcc) && (ismember(smoothDiffsAcc(a + i), peaksAcc) || ismember(smoothDiffsAlex(a + i), peaksAlex)) && (firstDiffAcc(a + i) < 0 || firstDiffAlex(a + i) < 0)
            if((smoothDiffsAcc(a + i) >= max(smoothDiffsAcc) * 0.25) || (smoothDiffsAlex(a + i) >= max(smoothDiffsAlex) * 0.25))
                bleach = true;
            end
        end
    end
end

