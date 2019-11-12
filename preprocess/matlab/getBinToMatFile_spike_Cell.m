
clear all; clc; 
%% Set paths -- this path is user dependant %
% utilities, data and output paths are passed
% Vision paths are defined in setPaths function
dataSet = '2015-11-09-3/';
toppath = '/nobackupkiwi/mstrange/CS230/';
dataPath = [toppath 'data/' dataSet 'data000/'];
%dataPath = '/nobackupkiwi/mstrange/CS230/data/2013-05-28-4/data000/';
outputPath = [toppath 'data/' dataSet 'mat/'];
%outputPath = '/nobackupkiwi/mstrange/CS230/data/2013-05-28-4/mat/';
outputPath1 = [toppath 'data/' dataSet 'compressed/'];
%outputPath1 = '/home/ubuntu/data/2013-05-28-4/data000/compressed/';
outputPath2 = [toppath 'data/' dataSet 'raw/'];
%outputPath2 = '/home/ubuntu/data/2013-05-28-4/data000/raw/';
util = [toppath 'vision7-unix/'];
%util = 'home/ubuntu/vision7-unix/';
spikePath= [toppath 'analysis/' dataSet 'data000/data000.spikes'];   
%spikePath='/home/ubuntu/analysis/2013-05-28-4/data000/data000.spikes';
setPaths(dataPath,outputPath,util);
spikeFile=edu.ucsc.neurobiology.vision.io.SpikeFile(spikePath);
%% Set global parameter
fs = 20e3;
Tmeas = 5*60;
nSamplesToRead = Tmeas*fs;   % how many samples to read
base_time_min = 10;
startSample = base_time_min * 60 * fs;
bufferSize = 100000; %1000000; % The number of samples to process at a time.
% The number of samples processed before starting a new .bin file
samplesPerBinFile = 2.4e6;  % 2.4e6 to limit to <2GB per file.

%% Start parallel pool
p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
    parpool(6)                 % leave 12 cores available
    nw = 6;
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

compressed_data = 0; % Set to 1 to use compressed

%% prepare data for compression
if exist(outputFile) && (compressed_data == 0)
    fprintf('Mat file already exists!\n');
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
        rawData = rawFile.getData(startSample, min(bufferSize,nSamplesToCopy));
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
        newData(startSample + 1 : startSample + size(tmp,1),:) = tmp;
        % You've copied a full buffer so you now need to copy less.
        nSamplesToCopy = nSamplesToCopy - bufferSize;
        startSample = startSample + bufferSize; % also increase your start index.
        disp(['====================== samples left to copy: ', num2str(nSamplesToCopy), ' ======================']);
    end
    disp([datestr(now, 'HH:MM:SS'),' -- Finished processing']);
    toc;
    % close the internal arrays to free memory.
    rawFile.close;
    %% uncomment line 84 and 85  to prepare file for pytorch to compress 
    save(outputFile, 'newData', '-v7.3');
    fprintf('Done creating output file\n');
else
    % Import compressed data 
    fprintf('Using the compressed input...\n');
    compressed_path = '/nobackupkiwi/mstrange/CS230/data/2015-11-09-3/mat/data_compressed_start0.csv';
    Compressed = csvread(compressed_path,0,0,[0 0 2000000 512]);
    fprintf('Finished reading the compressed input...\n');
end


%% Cropping the spikes
%eliminate electrode 45,49,177,401,418,460
%ArrayIndex=1;
samples_cropped = 0;
for ArrayIndex=1:1:512

    Tspike=spikeFile.getSpikeTimes(ArrayIndex);

    i = 1;
    start_spike = 10; % Always falls off the first one anyway...
    final_spike = 0;
    end_time_min = base_time_min + 5;
    if(base_time_min == 0) % Find the gap
        while i <= length(Tspike)
            if(Tspike(i) > (end_time_min * 60 * fs))
                final_spike = i - 100; % Just to be safe!
                break;
            end
            i = i + 1;
        end
    elseif (base_time_min == 25)
        final_spike = length(Tspike)
        while i <= length(Tspike)
            if(Tspike(i) > (base_time_min * 60 * fs))
                start_spike = i + 100; % Just to be safe!
                break;
            end
            i = i + 1;
        end
    else
        while i <= length(Tspike)
            if(Tspike(i) > (base_time_min * 60 * fs))
                start_spike = i + 100; % Just to be safe!
                break;
            end
            i = i + 1;
        end
        while i <= length(Tspike)
            if(Tspike(i) > (end_time_min * 60 * fs))
                final_spike = i - 100; % Just to be safe!
                break;
            end
            i = i + 1;
        end
    end

    fprintf('Processing electrode %d from %d minutes to %d minutes, using spikes %d to %d\n', ArrayIndex, base_time_min, end_time_min, start_spike, final_spike);
    spikeArr = zeros(72, (final_spike - start_spike + 1));

    len_comp = 0;
    if(compressed_data == 1)
        len_comp = length(Compressed(:, ArrayIndex));
        data_row = Compressed(:, ArrayIndex);
    else
        len_comp = length(newData(:, ArrayIndex));
        data_row = newData(:, ArrayIndex);
    end

    for i=start_spike:1:final_spike
        t=Tspike(i) - startSample;
        % compressed or raw...
        if((len_comp > t+59) && (t > 10))
            temp=data_row(t-10:t+60);
            spikeArr(:, i - start_spike + 1) = [i; temp];
        else
            fprintf('Fell of of the spike at %d\n', i);
        end
    end

    SpikeClip=[outputPath2 '/data000_electrode_' num2str(ArrayIndex) '_spikes_' num2str(start_spike) '_to_' num2str(final_spike) '.mat'];
    save(SpikeClip, 'spikeArr');

    samples_cropped = samples_cropped + ((final_spike - start_spike) * 71);
    disp(['====================== samples cropped: ', num2str(samples_cropped), ' ======================']);
 end

%  
% for ArrayIndex=1:1:512
%      Tspike=spikeFile.getSpikeTimes(ArrayIndex);
%      SpikeCount=size(Tspike);
%      SpikeCount=SpikeCount(1,1);
% %     for i=1:1:SpikeCount
%      for i=1:1:min(30,SpikeCount)
%          t=Tspike(i+1);
%           temp=newData(t-10:t+60,ArrayIndex);
%           SpikeClip=[outputPath '/data000_electrode_' num2str(ArrayIndex) '_spike_' num2str(i)  '.mat'];
%           save(SpikeClip, 'temp');
%           temp=newData(t+110:t+180,ArrayIndex);
%           SpikeClip=[outputPath '/data000_electrode_' num2str(ArrayIndex) '_nonspike_' num2str(i) '.mat'];
% 
%           save(SpikeClip, 'temp');
%      end
%      disp(['====================== samples cropped: ', num2str(ArrayIndex*30), ' ======================']);
%  end

%          temp=Compressed(t+110:t+180,ArrayIndex);
%          SpikeClip=[outputPath1 '/data000Compressed_electrode_' num2str(ArrayIndex) '_nonspike_' num2str(i) '.mat'];
%          save(SpikeClip, 'temp');
     %      temp=newData(t+110:t+180,ArrayIndex);
     %      SpikeClip=[outputPath2 '/data000_electrode_' num2str(ArrayIndex) '_nonspike_' num2str(i) '.mat'];
     %      save(SpikeClip, 'temp');
%comments: channel 412 has a nice spike as example
% figure(1)
% t = 1:1:size(rawData,1);
% tNew = size(newData,1) - size(rawData,1) + 1 : 1 : size(newData,1);
% nch = 1;
% %t = 550:1:700;
% plot(t,rawData(t,nch),t,newData(tNew:end,nch));
% 
% xlabel('sample');ylabel('code');
% grid on
% lg = legend('original','reconstructed','Location', 'southwest');
% set(lg,'FontSize',12)
% set(gcf,'color','w')
%  nyquistRate = B*20e3*nElec;
% mCollisions = mean(nCollisions)
% mActivityFree = mean(activityFactor)*20e3
% compressionRatio = nyquistRate/mActivityFree
  %  fprintf('Number of spikes for electrode %d: %d\n', ArrayIndex, length(Tspike));
  %  five_min = 5;
  %  i = 1;
  %  while i <= length(Tspike)
  %      if(Tspike(i) > (five_min * 60 * fs))
  %          fprintf('Crossed spike no %d at %d minutes\n', i, five_min);
  %          five_min =five_min + 5;
  %      end
  %      i = i + 1;
  %  end
%    ArrayIndex = ArrayIndex + 1;
 %   continue
 %     SpikeCount=size(Tspike);
%     SpikeCount=SpikeCount(1,1);
%                SpikeClip=[outputPath1 '/data000Compressed_electrode_' num2str(ArrayIndex) '_spike_' num2str(i)  '.mat'];
%                save(SpikeClip, 'temp');
%        else
%            if(mod(i, 1000) == 0)
%                fprintf('tic-tok');
%            end
%            if((len_raw > t+59) && (t > 10))
%                temp=newData(t-10:t+60,ArrayIndex);
%                spikeArr(:, i - start_spike + 1) = [i; temp];
%            else
%                fprintf('Ran out of spikes in this file at %d', i)
        %    end
        %end


