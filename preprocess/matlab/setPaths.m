function setPaths(dataPath,outputPath, util)
%% Set paths -- this path is user dependant %
visionPath = '/home/ubuntu/vision7-unix/Vision.jar';
visionWritePath = [util, 'WriteDataFile.jar'];
% Add Vision to the path if not already there.

javaaddpath(visionPath);
javaaddpath(visionWritePath);

if ~exist('edu/ucsc/neurobiology/vision/io/NeuronFile','class')
    error('ERROR - NeuronFile class cannot be found')
end

if ~exist('edu/ucsc/neurobiology/vision/io/ModifyRawDataFile','class')
    error('ERROR - ModifyRawDataFile class cannot be found')
end

if ~exist('edu/ucsc/neurobiology/vision/io/RawDataFile','class')
        error('ERROR - RawDataFile class cannot be found')
end

if ~exist('edu/ucsc/neurobiology/vision/electrodemap/ElectrodeMapFactory','class')
        error('ERROR - ElectrodeMapFactory class cannot be found')
end
