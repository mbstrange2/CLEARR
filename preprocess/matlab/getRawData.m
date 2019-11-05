function [rawFile, header, totalSamples, nElec] = getRawData(dataPath, nSamplesToRead, startSample)

% Ensure paths formatted.
if ( strcmp(dataPath(end), filesep) || strcmp(dataPath(end), '/') || ...
        strcmp(dataPath(end), '\') ), dataPath = dataPath(1:end-1); end

rawFile = edu.ucsc.neurobiology.vision.io.RawDataFile(dataPath);
header  = rawFile.getHeader();
totalSamples = header.getNumberOfSamples();
nElec   = header.getNumberOfElectrodes() - 1;
% If samples not specified, or too many
if(nSamplesToRead==0 || nSamplesToRead>totalSamples)
    nSamplesToRead = totalSamples - startSample;
end
header.setNumberOfSamples(nSamplesToRead);