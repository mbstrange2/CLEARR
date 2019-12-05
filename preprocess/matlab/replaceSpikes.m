clear all; clc; 
%% Set paths -- this path is user dependent %
% utilities, data and output paths are passed
% Vision paths are defined in setPaths function

% DATASET TO USE
dataSet = '2015-11-09-3/';
% BASE TIME IN MINUTES
base_time_min = 0; % Make sure to update
% TURN ON COMPRESSED DATA OR NOT
compressed_data = 0; % Set to 1 to use compressed

toppath = '/nobackupkiwi/mstrange/CS230/';
dataPath = [toppath 'data/' dataSet 'data000/'];
outputPath = [toppath 'data/' dataSet 'mat/'];
outputPath1 = [toppath 'data/' dataSet 'compressed/'];
outputPath2 = [toppath 'data/' dataSet 'raw/'];
util = [toppath 'vision7-unix/'];
spikePath= [toppath 'analysis/' dataSet 'data000/data000.spikes'];   
setPaths(dataPath,outputPath,util);
spikeFile=edu.ucsc.neurobiology.vision.io.SpikeFile(spikePath);

%% Set global parameter

fs = 20e3;
Tmeas = 10*60;
nSamplesToRead = Tmeas*fs;   % how many samples to read
startSample = base_time_min * 60 * fs;
bufferSize = 100000; %1000000; % The number of samples to process at a time.
% The number of samples processed before starting a new .bin file
samplesPerBinFile = 2.4e6;  % 2.4e6 to limit to <2GB per file.

%% Start parallel pool
p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
    parpool(2)                 % leave 12 cores available
    nw = 2;
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
outputFile = [outputPath '/data000_' num2str(Tmeas) '_sample_start_' num2str(base_time_min) 'm.mat'];

startSample_loop = startSample;
%% prepare data for compression
if exist(outputFile) && (compressed_data == 0)
    fprintf('Mat file already exists: %s\n', outputFile);
    data_file = matfile(outputFile);
    newData=data_file.newData;
    fprintf('newData read...\n');
elseif(compressed_data == 0)
    fprintf('Creating new mat file: %s\n', outputFile);
    % start with empty matrix
    newData = zeros(nSamplesToRead,nElec+1);
    while nSamplesToCopy > 0
     
        sampsThisFile = sampsThisFile + min(bufferSize,nSamplesToCopy);
     
        % either fills the buffer entirely, or grabs the remaining samples
        rawData = rawFile.getData(startSample_loop, min(bufferSize,nSamplesToCopy));
        nSamples = size(rawData,1);    
        if firstCycle
            % gets an offset per channel in a vector long nElec
            samplesForOffset = min(nSamples,50e3);
            [offsetPerChannel] = offsetPerChannel_mex(rawData(1:samplesForOffset,2:end));
            firstCycle = 0;
        end
  
        % define the tmp array and save calibrated data to it
        tmp = zeros(nSamples,nElec+1,'int16');
        parfor i=1:nSamples
            [tmp(i,:)] = offsetCalibration(rawData(i,:),offsetPerChannel);
        end
    
        % Appends data passed in to the final matrix
        newData(startSample_loop - startSample + 1 : startSample_loop - startSample + size(tmp,1),:) = tmp;
        % You've copied a full buffer so you now need to copy less.
        nSamplesToCopy = nSamplesToCopy - bufferSize;
        startSample_loop = startSample_loop + bufferSize; % also increase your start index.
        disp(['====================== samples left to copy: ', num2str(nSamplesToCopy), ' ======================']);
    end
    disp([datestr(now, 'HH:MM:SS'),' -- Finished processing']);
    toc;
    % close the internal arrays to free memory.
    rawFile.close;
    %% uncomment line 84 and 85  to prepare file for pytorch to compress 
    save(outputFile, 'newData', '-v7.3');
    fprintf('Done creating output file\n');
    exit;
end

num_replacements = 1
replacement_path = '/nobackupkiwi/mstrange/CS230/final_testing_spikes_';
for i=1:num_replacements
    temp_spikes = matfile([replacement_path num2str(i-1) '.mat']);
    if(i == 1)
        replacement_arr = temp_spikes.spikeRep;
    else
        replacement_arr = [replacement_arr, temp_spikes.spikeRep];
    end
end

% load in the replacement spikes...
% and make them into one array...

replacement_elecs = replacement_arr(:, 1);
replacement_spikes = replacement_arr(:, 2);

startSample = base_time_min * 60 * fs; % Put this back in for offsetting
%% Cropping the spikes
%eliminate electrode 45,49,177,401,418,460
for ArrayIndex=1:1:512

    fprintf('Electrode Number: %d\n', ArrayIndex);
    % Get spike times for this electrode
    Tspike=spikeFile.getSpikeTimes(ArrayIndex);

    %
    %i = 1;
    %start_spike = 1; % Always falls off the first one anyway...
    %final_spike = 0;
    %end_time_min = base_time_min + 10;
    % Get the spikes that are in here...
    %while i <= length(Tspike)
    %    if(Tspike(i) > (end_time_min * 60 * fs))
    %        final_spike = i; % Just to be safe!
    %        break;
    %    end
    %    i = i + 1;
    %end

    % Now look for the spike in the replacement array...
    electrode = ArrayIndex;

    % Find the elements in replacement array who refer to electrode
    found_elec = find(replacement_elecs == electrode);
    data_row = newData(:, electrode);
    
    for qq=1:1:size(found_elec)
        spike = replacement_spikes(found_elec(qq));
        spike_time = Tspike(spike);
        data_row(spike_time-10 : spike_time+60) = replacement_arr(found_elec(qq), 3:end);
    end

    newData(:, electrode) = data_row;

    %len_comp = 0;
   % if(compressed_data == 1)
   %     len_comp = length(Compressed(:, ArrayIndex));
     %   data_row = Compressed(:, ArrayIndex);
   % else 
   %     len_comp = length(newData(:, ArrayIndex));
   %     data_row = newData(:, ArrayIndex);
   % end

    %for i=start_spike:1:final_spike
    %    t=Tspike(i) - startSample;
    %    % compressed or raw...
    %    if((len_comp > t+59) && (t > 10))
    %        temp=data_row(t-10:t+60);
    %        spikeArr(:, i - start_spike + 1) = [i; temp];
    %    else
    %        fprintf('Fell of of the spike at %d\n', i);
    %    end
    %end 
    %

end

full_replaced=[outputPath '/data000_replaced.mat'];
save(full_replaced, 'newData', '-v7.3');


