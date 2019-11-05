%% Author: Dante Muratore - dantemur@stanford.edu
% 05-Feb-2018
% -- based on previous work from Nathan Staffa
%
% This function provides basic Matlab interaction to .bin files from
% retina recordings in the Chichilnisky lab. It uses the spike sorter
% Vision to support some of the functionalities.

clear all;close all;clc; 

%% Set paths -- this path is user dependant %
% utilities, data and output paths are passed
% Vision paths are defined in setPaths function
dataPath = '/Volumes/Scratch/Users/dantemur/data/2015-11-09-3/orig/data000';
dataToModifyPath = '/Volumes/Scratch/Users/dantemur/data/2015-11-09-3/lossy_1ch_interp/data000';
outputPath = '/Volumes/Scratch/Users/dantemur/data/2015-11-09-3/bin/data000';
util = '/Volumes/Lab/Users/dantemur/readout/matlab/utilities/';
setPaths(dataPath,outputPath,util)
%% Set global parameters
nSamplesToRead = 0;   % how many samples to read
startSample = 0;
bufferSize = 100000; % The number of samples to process at a time.
% The number of samples processed before starting a new .bin file
samplesPerBinFile = 2.4e6;  % 2.4e6 to limit to <2GB per file.
B = single(12);         % B bits for quantizer
FS = single(2048);      % original full scale
FSeff = single(256);       % adpated to maximum signal
Beff = single(B - ceil(log2(FS/FSeff)));    % effective number of bits
cF = single(8);     % compression factor for data stream


%% Get the raw file handler object and electrode map
disp('=============================== Initializing Algorithms ===============================');
% Gets raw data and sets header
rawFile = edu.ucsc.neurobiology.vision.io.RawDataFile(dataPath);
modifyFile = edu.ucsc.neurobiology.vision.io.RawDataFile(dataToModifyPath);
header  = rawFile.getHeader();
totalSamples = header.getNumberOfSamples();
nElec   = header.getNumberOfElectrodes() - 1;
% If samples not specified, or too many
if(nSamplesToRead==0 || nSamplesToRead>totalSamples)
    nSamplesToRead = totalSamples - startSample;
end
header.setNumberOfSamples(nSamplesToRead);
% Gets electrode map to move from 512 vector to 16x32 array
getMap
%% Runs readout + decoder on 100'000 samples at a time
fprintf([datestr(now, 'HH:MM:SS'),'\n -- Processing data from: ', dataPath,...
    '\n -- Saving data to: ', outputPath, '\n']);
tic;
% Stores the number of samples remaining to be copied.
nSamplesToCopy = nSamplesToRead; % So it resets each time.
% Create the new file (Creates dataxyz000.bin and copies the header info)
newFile = edu.ucsc.neurobiology.vision.io.ModifyRawDataFile(outputPath, header);
% Keeps track of the current file's # of samples to allow of ensuring that
% the individual files stay under 2GB (in case of old FAT filesystems).
sampsThisFile = 0;
%parpool(8);     % sets maximum number of cores for parfor
rng(0,'twister');% sets the seed for predictability

while nSamplesToCopy > 0
    
    sampsThisFile = sampsThisFile + min(bufferSize,nSamplesToCopy);
    if (sampsThisFile > samplesPerBinFile) % We're at the size limit
        newFile.addFile; % Start a new .bin file
        sampsThisFile = 0;
    end
    
    % either fills the buffer entirely, or grabs the remaining samples
    rawData = rawFile.getData(startSample, min(bufferSize,nSamplesToCopy));
    modifyData = modifyFile.getData(startSample, min(bufferSize,nSamplesToCopy));
    % copies synchronization channel
    modifyData(:,1) = rawData(:,1);
    % Appends data passed in to the end of final .bin file in the folder
    newFile.appendDataToLastFile(modifyData);
    % You've copied a full buffer so you now need to copy less.
    nSamplesToCopy = nSamplesToCopy - bufferSize;
    startSample = startSample + bufferSize; % also increase your start index.
    disp(['====================== samples left to copy: ', num2str(nSamplesToCopy), ' ======================']);
end

disp([datestr(now, 'HH:MM:SS'),' -- Finished processing']);
toc;

% newFile.close; % close the internal arrays to free memory.
rawFile.close;

%%

%comments: channel 412 has a nice spike as example
figure(1)
t = 1:1:size(rawData,1);
%t = 550:1:700;
plot(t,rawData(t,412),t,modifyData(t,412));

xlabel('sample');ylabel('code');
grid on
lg = legend('original','reconstructed','Location', 'southwest');
set(lg,'FontSize',12)
set(gcf,'color','w')
% 
% nyquistRate = B*20e3*nElec;
% mCollisions = mean(nCollisions)
% mActivityFree = mean(activityFactor)*20e3
% compressionRatio = nyquistRate/mActivityFree