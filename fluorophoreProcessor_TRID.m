clear all
close all

CDir = cd;
fileNames = dir('*.mat');
cd(CDir);

% for i = 1:length(fileNames)
while true
    
    folderName = input("Which folder would you like to open (TCD1, TCD2, TCR1, TCR2): ", "s");

    % base case to break the infinite while loop
    if strcmp(folderName, "Quit") || strcmp(folderName, "quit") || strcmp(folderName, "Q") || strcmp(folderName, "q")
        disp("Quitting this program . . .");
        break;
    end

    fileNum = input("Which file number would you like to open: ", "s");
    fileName = getFileName(folderName, fileNum);
    disp("Opening file " + fileName + " . . . ");

    data = load(fileName); % load(fileNames(i).name);
    data = data.data;

    % smooth and denoise the data
    col1 = preprocess(data(:,1));
    col2 = preprocess(data(:,2));
    col3 = preprocess(data(:,3));

    % calculate negative derivatives
    firstDiff1 = -diff(col1).^2;
    firstDiff2 = diff(col2);
    firstDiff3 = diff(col3);

    % generates plot
    figure('Renderer', 'painters', 'Position', [1 1 1500 1500])
    
    a = subplot(2,1,1);
    hold all
    plot(data(:,1), 'color', 'g');
    plot(col1, 'color', 'b', 'LineWidth', 5)
    title('Comparison Between Original tRNA and Smoothed/Denoised tRNA Emission Intensities')
    xlabel 'Time (s)';
    ylabel 'Emission Intensity (J)';
    legend('tRNA (Donor)', 'Preproccesed tRNA')
    set(gca, 'fontsize', 18)

    b = subplot(2,1,2);
    plot(firstDiff1, 'color', 'b', 'LineWidth', 5);
    title('Derivative of Preprocessed tRNA (Donor) Emission Intensities')
    xlabel 'Time (s)';
    ylabel 'Emission Intensity per Second (J/s)';
    legend('tRNA (Donor)')
    set(gca, 'fontsize', 18)
    grid on
    linkaxes([a,b],'x')
end

% identifies the name of the ".mat" file that each numbered file belongs to
function fName = getFileName(folder, num)
    if strcmp(folder, "TCD1")
        prefix = "0204";
    elseif strcmp(folder, "TCD2")
        prefix = "0207";
    elseif strcmp(folder, "TCR1")
        prefix = "0201";
    else % folder = "TCR2"
        prefix = "0203";
    end

    fNum = "";
    iterNum = 3 - strlength(num);
    for i = 1:iterNum
        fNum = fNum + "0";
    end

    fNum = fNum + num;
    fName = "s" + prefix + "-wr.tiff-pairProfile" + fNum + "-sb-tc.mat";
end


