%% Author: Dante Muratore - dantemur@stanford.edu
% 05-Feb-2018
% -- based on previous work from Nathan Staffa
%
% This script provides basic Matlab interaction to .bin files from
% retina recordings in the Chichilnisky lab. It uses the spike sorter
% Vision to support some of the functionalities.

% The script reads .bin files, calibrate the offset of each channel and
% saves the output in a .mat file.

clear all;close all;clc; 

%% Set paths -- this path is user dependant %
% utilities, data and output paths are passed
% Vision paths are defined in setPaths function
dataPath = '/sim/dantemur/readout/data/2015-11-09-3/orig/data000/';
outputPath = '/sim/dantemur/readout/data/2015-11-09-3/orig_6min/data000';
util = '/sim/dantemur/readout/utilities/';
setPaths(dataPath,outputPath,util)
%% Set global parameters
fs = 20e3;
Tmeas = 60*6;
nSamplesToRead = Tmeas*fs;   % how many samples to read
startSample = 0;
bufferSize = 1000000; % The number of samples to process at a time.
% The number of samples processed before starting a new .bin file
samplesPerBinFile = 2.4e6;  % 2.4e6 to limit to <2GB per file.

%% Start parallel pool
p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
    parpool(16)                 % leave 12 cores available
    nw = 16;
    fprintf(['Local cluster built with ', num2str(nw), ' workers.\n'])
else
    nw = p.NumWorkers;
    fprintf(['Local cluster already exists with ', num2str(nw), ' workers.\n'])
end
%% Get the raw file handler object
disp('=============================== Initializing Algorithms ===============================');
% Gets raw data and sets header
[rawFile, header, totalSamples, nElec] = getRawData(dataPath, nSamplesToRead, startSample);
%% Runs readout + decoder on 100'000 samples at a time
fprintf([datestr(now, 'HH:MM:SS'),'\n -- Processing data from: ', dataPath,...
    '\n -- Saving data to: ', outputPath, '\n']);
tic;
% Stores the number of samples remaining to be copied.
nSamplesToCopy = nSamplesToRead; % So it resets each time.
% Keeps track of the current file's # of samples to allow of ensuring that
% the individual files stay under 2GB (in case of old FAT filesystems).
sampsThisFile = 0;
firstCycle = 1;
% Create the new file (Creates dataxyz000.bin and copies the header info)
newFile = edu.ucsc.neurobiology.vision.io.ModifyRawDataFile(outputPath, header);

while nSamplesToCopy > 0
    
    sampsThisFile = sampsThisFile + min(bufferSize,nSamplesToCopy);
    if (sampsThisFile > samplesPerBinFile) % We're at the size limit
        newFile.addFile; % Start a new .bin file
        sampsThisFile = 0;
    end
    
    % either fills the buffer entirely, or grabs the remaining samples
    rawData = rawFile.getData(startSample, min(bufferSize,nSamplesToCopy));
    nSamples = size(rawData,1);
    if firstCycle
        % gets an offset per channel in a vector long nElec
        samplesForOffset = min(nSamples,50e3);
        [offsetPerChannel] = offsetPerChannel_mex(rawData(1:samplesForOffset,2:end));
        firstCycle = 0;
    end
    
    % define the tmp array and save calibrated data to it
    newData = zeros(nSamples,nElec+1,'int16');
    parfor i=1:nSamples
        [newData(i,:)] = offsetCalibration(rawData(i,:),offsetPerChannel);
    end
    
    % Appends data passed in to the end of final .bin file in the folder
    newFile.appendDataToLastFile(newData);
    
    % You've copied a full buffer so you now need to copy less.
    nSamplesToCopy = nSamplesToCopy - bufferSize;
    startSample = startSample + bufferSize; % also increase your start index.
    disp(['====================== samples left to copy: ', num2str(nSamplesToCopy), ' ======================']);
end

disp([datestr(now, 'HH:MM:SS'),' -- Finished processing']);
toc;

newFile.close; % close the internal arrays to free memory.
rawFile.close;

%%

%comments: channel 412 has a nice spike as example
figure(1)
t = 1:1:size(rawData,1);
tNew = size(newData,1) - size(rawData,1) + 1 : 1 : size(newData,1);
nch = 1;
%t = 550:1:700;
plot(t,rawData(t,nch),t,newData(tNew:end,nch));

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