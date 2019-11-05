%% re-convert data with ramp-ADC architecture
function [dataOut] = offsetCalibration(dataIn, offsetPerChannel)
dataOut = int16(zeros(size(dataIn)));

% subtract offset per channel
dataOut(2:end) = dataIn(2:end)-offsetPerChannel;
dataOut(1) = dataIn(1);
