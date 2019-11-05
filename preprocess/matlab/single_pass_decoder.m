%% re-convert data with ramp-ADC architecture
function [dataOut] = single_pass_decoder(dataIn, posLin, B, FS)
% variable definitions and initialization
arrayIn = int16(zeros(16,32));
arrayOut = int16(zeros(16,32));
arrayDecod = int16(zeros(16,32));
nCodes = int16(max(max(abs(dataIn(2:end)))));
row = int16(zeros(16,1));
col = int16(zeros(1,32));
validOut = int16(ones(16,32));
dataOut = int16(zeros(size(dataIn)));
i = int16(1);
ramp = int16(zeros(2^B,1));

%% test
% display('running adc_ramp in test mode')
% dataIn = rawData(i,:);

%% processing
arrayIn(posLin) = dataIn(2:end);
% linear ramp
ramp = (-FS:2*FS/2^B:FS)+0.5;
% half sine ramp
%ramp = FS*sin(-pi/2:pi/2^B:pi/2);

for i = 1:size(ramp,2)-1
    % find row and col vector per code per sample
    row = int16(sum((arrayIn(:,:)>ramp(i))&(arrayIn(:,:)<=ramp(i+1)),2)>0);
    col = int16(sum((arrayIn(:,:)>ramp(i))&(arrayIn(:,:)<=ramp(i+1)),1)>0);
    % create a decoded array (cross row and column)
    % avoid overwriting strong decisions - no conflict in making them
    arrayDecod=row*col.*validOut;
    % overwrite the output array with the decoded array
    arrayOut = (arrayOut.*abs(arrayDecod-1)) + arrayDecod*ramp(i);
    % update the strong decision mask for each code looking at tmp
    validOut = validOut - arrayDecod * single(min(sum(row==1,1),sum(col==1,2)) == 1);
end

% move from matrix of channels to vector of channels (use ElectrodeMap)
dataOut(2:end) = arrayOut(posLin);
dataOut(1) = dataIn(1);