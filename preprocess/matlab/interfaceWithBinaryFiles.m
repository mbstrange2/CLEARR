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
dataPath = '/Volumes/Data/2015-11-09-3/data000';
outputPath = '/Volumes/Scratch/Users/dantemur/data/2015-11-09-3/bin/data000';
util = '/Volumes/Lab/Users/dantemur/readout/matlab/utilities/';
setPaths(dataPath,outputPath,util)


%% Set global parameters

nSamplesToRead = uint32(20000);   % how many samples to read
startSample = uint32(0);
bufferSize = uint32(100000); % The number of samples to process at a time.
% The number of samples processed before starting a new .bin file
samplesPerBinFile = uint32(2.4e6);  % 2.4e6 to limit to <2GB per file.
B = uint16(12);         % B bits for quantizer
FS = uint16(2048);      % original full scale
FSeff = uint16(256);       % adpated to maximum signal
Beff = B - uint16(ceil(log2(single(FS/FSeff))));    % effective number of bits
cF = uint16(8);     % compression factor for data stream

%% Get the raw file handler object
disp('=============================== Initializing Algorithms ===============================');

% Gets raw data and sets header
getRawData
%% Create Electrode Map
% The ElectrodeMapFactory object handles the construction of ElectrodeMap objects
% from unique data set identifiers contained in the raw data header
elMapFactory = edu.ucsc.neurobiology.vision.electrodemap.ElectrodeMapFactory();
% Use the electrode map factory and array ID to get Vision to tell you what the electrode map was for the recording
elMap = elMapFactory.getElectrodeMap(header.getArrayID());
coordinates = elMap.toString;
elMap_dir = [util 'elMap.txt'];
x_offset = ones(nElec,2)*[16.25 8; 16.25 8];
posPhy = table2array(readtable(elMap_dir,'ReadVariableNames',false));
posLog = floor((posPhy./30.+x_offset)./2.+0.5);
posLog(:,[1 2]) = posLog(:,[2 1]);
posLin = single(sub2ind([16,32],posLog(:,1),posLog(:,2)));

%%

fprintf([datestr(now, 'HH:MM:SS'),'\n -- Processing data from: ', dataPath,...
    '\n -- Saving data to: ', outputPath, '\n']);
tic;

% Stores the number of samples remaining to be copied.
nSamplesToCopy = nSamplesToRead; % So it resets each time.

% Create the new file (Creates dataxyz000.bin and copies the header info)
newFile = edu.ucsc.neurobiology.vision.io.ModifyRawDataFile(outputPath, header);

% Keeps track of the current file's # of samples to allow of
% ensuring that the individual files stay under 2GB (in case of old
% FAT filesystems).
sampsThisFile = 0;

while nSamplesToCopy > 0
    
    sampsThisFile = sampsThisFile + min(bufferSize,nSamplesToCopy);
    if (sampsThisFile > samplesPerBinFile) % We're at the size limit
        newFile.addFile; % Start a new .bin file
        sampsThisFile = 0;
    end
    
    % either fills the buffer entirely, or grabs the remaining samples
    rawData = rawFile.getData(startSample, min(bufferSize,nSamplesToCopy));
    nSamples = size(rawData,1);   
    % define the newData array as int16
    newData = zeros(nSamples,nElec+1,'int16');
    % remove per channel offset
    rawData(:,2:end) = rawData(:,2:end)-repmat(round(mean(rawData(:,2:end),1,'native')),nSamples,1);

    % Samples are run in parallel for speed
    % Each sample is processed by a compiled function (ramp_adc) for speed
    parfor i=1:nSamples
        [newData(i,:)] = single_pass_decoder(rawData(i,:),posLin,Beff,FSeff);
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