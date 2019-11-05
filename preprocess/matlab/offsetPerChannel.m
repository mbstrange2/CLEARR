function [offsetPerChannel] = offsetPerChannel(data)
nSamples = int16(0);
nSamplesPerOffset = int16(0);
offsetPerChannel = zeros(1,512,'int16');

nSamples = size(data,1);
nSamplesPerOffset = min(nSamples,5000);
offsetPerChannel = int16(round(mean(data(1:nSamplesPerOffset,:),1)));
