%% Set paths

% Path to Vision.jar
visionPath = '/home/dantemur/retina/Java/vision7/Vision.jar';
javaaddpath(visionPath);
% Path to test
% rawDataDir = uigetdir('/Users/dantemur/Desktop/retina/degrading tests/', 'Select raw data directory'); 
testsPath = '/home/dantemur/Desktop/tmp/2016-06-13-3/';
rawDataPath = [testsPath 'data000'];
neuronPath = [testsPath 'data000/data000.neurons'];
eiPath = [testsPath 'data000/data000.ei'];

%% Get Electrode Map

% Get the array ID for an example raw data file
rawDataFile = edu.ucsc.neurobiology.vision.io.RawDataFile(rawDataPath);
rawDataHeader = rawDataFile.getHeader();

% The ElectrodeMapFactory object handles the construction of ElectrodeMap objects 
% from unique data set identifiers contained in the raw data header
elMapFactory = edu.ucsc.neurobiology.vision.electrodemap.ElectrodeMapFactory();

% Use the electrode map factory and array ID to get Vision to tell you what the electrode map was for the recording
elMap = elMapFactory.getElectrodeMap(rawDataHeader.getArrayID());
coordinates = elMap.toString;
elMap.getNumberOfElectrodes;
% fid = fopen('elMap.txt','wt');
% fprintf(fid, '%s', coordinates);
% fclose(fid);
pos = table2array(readtable('elMap.txt'));

newPos = zeros(size(pos));
for col = 1:2
    posListSort = sort(unique(pos(:,col)));
    for newInd = 1:length(posListSort)
        prevInd = posListSort(newInd);
        newPos(pos(:,col)==prevInd,col) = newInd;  % logical indexing
    end
end
newPos = newPos';

%% Get Neurons EIs and electrode for maximum spike in each neuron

% Get Neuron IDs
neuronFile = edu.ucsc.neurobiology.vision.io.NeuronFile(neuronPath);
IDs = neuronFile.getIDList();

% The PhysiologicalImagingFile object handles the Electrical Images of all
% the Neurons found in the test
eiFile = edu.ucsc.neurobiology.vision.io.PhysiologicalImagingFile(eiPath);

% Pre-instance matrix for EIs and read one neuron at a time
neuronEI_volt = zeros(neuronFile.getNumberOfNeurons, eiFile.nElectrodes, eiFile.nSamples);
for i = 1 : neuronFile.getNumberOfNeurons
    ei_tmp = eiFile.getImage(IDs(i));
    neuronEI_volt(i,:,:) = squeeze(ei_tmp(1,:,:));
end

%% Plot histogram of principal spike amplitude per neuron

% Find electrode for maximum spike in each neuron
for i = 1 : neuronFile.getNumberOfNeurons
    current_neuron = abs(squeeze(neuronEI_volt(i,:,:)));
    [ampl(i), elec(i)] = max(max(current_neuron'));
end


% Plot histogram
figure()
bins = 10 : 05 : 1000;
hist(4/2^12*ampl/270*1e6, bins)
title('Principal spike amplitude per neuron')
xlabel('Principal spike amplitude [uV]')
ylabel('Number of neurons')

%% Plot map of maximum spike per electrode
%% Get Neurons EIs and electrode for maximum spike in each neuron

% Get Neuron IDs
neuronFile = edu.ucsc.neurobiology.vision.io.NeuronFile(neuronPath);
IDs = neuronFile.getIDList();

% The PhysiologicalImagingFile object handles the Electrical Images of all
% the Neurons found in the test
eiFile = edu.ucsc.neurobiology.vision.io.PhysiologicalImagingFile(eiPath);

% Pre-instance matrix for EIs and read one neuron at a time
neuronEI_volt = zeros(neuronFile.getNumberOfNeurons, eiFile.nElectrodes, eiFile.nSamples);
for i = 1 : neuronFile.getNumberOfNeurons
    ei_tmp = eiFile.getImage(IDs(i));
    neuronEI_volt(i,:,:) = squeeze(ei_tmp(1,:,:));
end

%% Plot histogram of principal spike amplitude per neuron

% Find electrode for maximum spike in each neuron
for i = 1 : neuronFile.getNumberOfNeurons
    current_neuron = abs(squeeze(neuronEI_volt(i,:,:)));
    [ampl(i), elec(i)] = max(max(current_neuron'));
end
% Find maximum spike per electrode



C = zeros(max(newPos'))';
for i = 1: neuronFile.getNumberOfNeurons
    xpos = newPos(2,elec(i));
    ypos = newPos(1,elec(i));
    if ampl(i)>C(xpos,ypos)
        C(xpos,ypos) = ampl(i);
    end
end

C = C./max(C)*64;

IM = image(C);


figure()
plot(squeeze(neuronEI_volt(123,107,:))')

    