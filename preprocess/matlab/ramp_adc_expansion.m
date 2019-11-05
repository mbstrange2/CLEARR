%% re-convert data with ramp-ADC architecture
function [dataOut] = ramp_adc_expansion(dataIn, offsetPerChannel, posLin, B, FS, nc)
% variable definitions and initialization
arrayIn = int16(zeros(16,32));
arrayOut = single(zeros(16,32));
arrayDecod = single(zeros(16,32));
nCodes = int16(max(max(abs(dataIn(2:end)))));
row = single(zeros(16,1));
col = single(zeros(1,32));
validOut = single(ones(16,32));
dataOut = int16(zeros(size(dataIn)));
i = logical(1);
ramp = single(zeros(2^B,1));
symbolsDecoded = single(0);

%% processing
arrayIn(posLin) = dataIn(2:end)-offsetPerChannel;
% linear ramp
ramp = (-FS:2*FS/2^B:FS);
% half sine ramp
%ramp = FS*sin(-pi/2:pi/2^B:pi/2);

for i = 1:size(ramp,2)-1
    % find row and col vector per code per sample
    row = single(sum((arrayIn(:,:)>ramp(i))&(arrayIn(:,:)<=ramp(i+1)),2)>0);
    col = single(sum((arrayIn(:,:)>ramp(i))&(arrayIn(:,:)<=ramp(i+1)),1)>0);
    % save values only if #collisions less than nc
    symbolsDecoded = sum(row==1,1)*sum(col==1,2);
    if symbolsDecoded <= nc
        % create a decoded array (cross row and column)
        % avoid overwriting strong decisions - no conflict in making them
        arrayDecod=row*col.*validOut;
    else
        arrayDecod = single(zeros(16,32));
    end
    % overwrite the output array with the decoded array
    arrayOut = (arrayOut.*abs(arrayDecod-1)) + arrayDecod*ramp(i);
    % update the strong decision mask for each code looking at tmp
    validOut = validOut - arrayDecod * single(min(sum(row==1,1),sum(col==1,2)) == 1);
end

% move from matrix of channels to vector of channels (use ElectrodeMap)
dataOut(2:end) = cast(arrayOut(posLin),'int16');
dataOut(1) = dataIn(1);