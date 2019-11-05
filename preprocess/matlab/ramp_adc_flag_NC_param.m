%% re-convert data with ramp-ADC architecture
function [dataOut, bitTx] = ramp_adc_flag_NC_param(dataIn, offsetPerChannel, posLin, B, FS, nwire)
% variable definitions and initialization
arrayIn = int16(zeros(16,32));
arrayOut = single(zeros(16,32));
arrayDecod = single(zeros(16,32));
mask = int16(zeros(16,32));
nCodes = int16(max(max(abs(dataIn(2:end)))));
row = single(zeros(16,1));
col = single(zeros(1,32));
validOut = single(ones(16,32));
dataOut = int16(zeros(size(dataIn)));
i = logical(1);
ramp = single(zeros(2^B,1));
notConflict = logical(0);
bitTx = single(0);

%% test
% display('running adc_ramp in test mode')
% dataIn = rawData(41,:);

%% processing
arrayIn(posLin) = dataIn(2:end)-offsetPerChannel;
LSB = (2*FS/2^B);
minValue = single(min(min(arrayIn)));
maxValue = single(max(max(arrayIn)));
VRn = (floor(minValue/LSB) * LSB) - LSB;
VRp = (ceil(maxValue/LSB) * LSB) + LSB;
% linear ramp
ramp = (VRn:LSB:VRp);
%ramp = (-FS:2*FS/2^B:FS);
% half sine ramp
%ramp = FS*sin(-pi/2:pi/2^B:pi/2);
% 

for i = 1:size(ramp,2)-1
    % arrayDecod is accumulated across nwire - so it has to be
    % reset each time
    arrayDecod = single(zeros(16,32));
    for j = 1:nwire
        % mask provides the right channels to each decoder
        mask(j:nwire:end) = 1;
        % find row and col vector per code per sample
        row = single(sum((arrayIn.*mask>ramp(i))&(arrayIn.*mask<=ramp(i+1)),2)>0);
        col = single(sum((arrayIn.*mask>ramp(i))&(arrayIn.*mask<=ramp(i+1)),1)>0);
        % save values only if notConflict
        notConflict = min(sum(row==1,1),sum(col==1,2)) == 1;
        % create a decoded array (cross row and column)
        % avoid overwriting strong decisions - no conflict in making them
        arrayDecod = row*col.*validOut*notConflict + arrayDecod;
        % reset mask for next decoder
        mask = int16(zeros(16,32));
    end
    % overwrite the output array with the decoded array
    %             arrayOut = (arrayOut.*abs(arrayDecod-1)) + arrayDecod*ramp(i+1);
    arrayOut = arrayOut + arrayDecod*ramp(i+1);
    % update the strong decision mask for each code looking at tmp
    validOut = validOut - arrayDecod * single(min(sum(row==1,1),sum(col==1,2)) == 1);
end
% counts number of flags and multiplies for the address word
bitTx = sum(sum(validOut == 0)) * 9;

% move from matrix of channels to vector of channels (use ElectrodeMap)
dataOut(2:end) = cast(arrayOut(posLin),'int16');
dataOut(1) = dataIn(1);
